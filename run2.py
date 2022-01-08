from typing import Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import msssim
from guass import *
from torch.autograd import Variable as V
import math
import scipy.stats as st
class Attacker2:
    def __init__(self,
                 steps: int,
                 quantize: bool = None,#取整
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 THRESHOLD:Optional[float] = None,
                 div_prob: float = 0.9,
                 loss_amp: float = 4.0,
                 alpha:float=1.0/255.0,
                 device: torch.device = torch.device('cuda:0')) -> None:
        self.steps = steps

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.div_prob = div_prob
        self.loss_amp = loss_amp
        self.THRESHOLD=THRESHOLD
        self.alpha=alpha

        self.device = device

    def get_kernel(self,kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def TI_kernel(self):
        kernel_size = 5  # kernel size
        kernel = self.get_kernel(kernel_size, 1).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
        return gaussian_kernel

    # gaussian_kernel for filter high frequency information of images
    def gaussian_kernel(self,device, kernel_size=15, sigma=2, channels=3):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()  # kernel_size*kernel_size*2
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, padding=(kernel_size - 1) // 2,
                                    bias=False)
        gaussian_filter.weight.data = gaussian_kernel.to(device)
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    # TV loss
    def tv_loss(self,input_t):
        temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
        temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        return temp.sum(axis=1).sum(axis=1).sum(axis=1)

    def get_init_noise(self,device=torch.device("cuda:0")):
        noise = torch.Tensor(8, 3, 112, 112)
        noise = torch.nn.init.xavier_normal(noise, gain=1)
        return noise.to(device)

    def input_diversity(self,x, resize_rate=1.15, diversity_prob=0.7):
        assert resize_rate >= 1.0
        assert diversity_prob >= 0.0 and diversity_prob <= 1.0
        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        ret = padded if torch.rand(1) < diversity_prob else x
        return ret

    def attack(self,
               model: nn.Module,
               inputs: torch.Tensor,
               inputs_tar:torch.Tensor,
               # mask:torch.Tensor,
               labels_true: torch.Tensor,
               labels_target: torch.Tensor
               ) -> torch.Tensor:
        num_iter = self.steps
        batch_size = inputs.shape[0]
        n=7
        delta = torch.zeros_like(inputs, requires_grad=True).to(self.device)
        optimizer = optim.SGD([delta], lr=0.5, momentum=0.9)

        # setup optimizer
        # optimizer = optim.SGD([delta], lr=1, momentum=0.9)
        #
        # # for choosing best results
        best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=self.device)
        best_delta = torch.zeros_like(inputs)
        seed_num = 0  # set seed
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.backends.cudnn.deterministic = True

        # gaussian_kernel: filter high frequency information of images
        gaussian_smoothing = self.gaussian_kernel(self.device, kernel_size=5, sigma=1, channels=3)
        # X = gaussian_smoothing(inputs).to(self.device)  # filter high frequency information of images
        X = (inputs).to(self.device)
        logits_target = model(inputs_tar).clone().detach()
        logits_true = model(inputs).clone().detach()
        v=torch.zeros_like(inputs).detach().to(self.device)
        for i in range(num_iter):
            g_temp = []

            for i in range(n):
                # 进行n次input_diversity
                X_adv = X + delta
                X_adv = self.input_diversity(X_adv)
                X_adv = F.interpolate(X_adv, (112, 112), mode='bilinear', align_corners=False)
                # get ensemble logits
                logits1 = model(X_adv)
                loss11 = F.mse_loss(logits1, logits_target, reduction='none').sum(axis=1)
                loss12 = F.mse_loss(logits1, logits_true, reduction='none').sum(axis=1)
                loss1 = 4 * loss11 - loss12
                loss1 = torch.mean(loss1)
                loss1.backward()
                grad = delta.grad.clone()
                # TI: smooth the gradient
                # grad = F.conv2d(grad, self.TI_kernel(), bias=None, stride=1, padding=(2, 2), groups=3)
                g_temp.append(grad)

            g=torch.zeros_like(inputs).detach().to(self.device)
            for j in range(n):
                g += g_temp[j]
            g = g / n
            delta.grad.zero_()
            # X_adv = X + delta
            # logits1 = model(X_adv)
            # loss11 = F.mse_loss(logits1, logits_target, reduction='none').sum(axis=1)
            # loss12 = F.mse_loss(logits1, logits_true, reduction='none').sum(axis=1)
            # loss1 = 4 * loss11 - loss12
            # loss=loss1
            # is_better = loss < best_loss
            # best_loss[is_better] = loss[is_better]
            # best_delta[is_better] = delta.data[is_better]
            # loss = torch.mean(loss)
            # optimizer.zero_grad()
            # loss.backward()

            grad_norms = g.view(batch_size, -1).norm(p=float('inf'), dim=1)
            g.div_(grad_norms.view(-1, 1, 1, 1))
            # delta.grad=0.1*delta.grad+g
            v=0.9*v+0.5*g
            delta.data=delta.data-v
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                g[grad_norms == 0] = torch.randn_like(g[grad_norms == 0])
            #无穷范数攻击
            # delta.data = delta.data - self.alpha * torch.sign(g)
            delta.data = delta.data.clamp(- self.max_norm,  self.max_norm)
            delta.data = ((inputs + delta.data).clamp(0.0, 1.0)) - inputs
        return delta+inputs
    # return best_delta
