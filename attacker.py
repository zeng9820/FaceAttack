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
import scipy.stats as st
import math
class Attacker:
    def __init__(self,
                 steps: int,
                 quantize: bool = None,#取整
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 THRESHOLD:Optional[float] = None,
                 div_prob: float = 0.9,
                 loss_amp: float = 4.0,
                 device: torch.device = torch.device('cuda:0')) -> None:
        self.steps = steps

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.div_prob = div_prob
        self.loss_amp = loss_amp
        self.THRESHOLD=THRESHOLD

        self.device = device

    def get_kernel(self, kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def TI_kernel(self):
        kernel_size = 3  # kernel size
        kernel = self.get_kernel(kernel_size, 1).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
        return gaussian_kernel

    # gaussian_kernel for filter high frequency information of images
    def gaussian_kernel(self, device, kernel_size=15, sigma=2, channels=3):
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
    def inf_loss(self,inputs):#将三个通道像素差异的最大值相加
        b = torch.max(torch.max(inputs, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0].reshape(inputs.shape[0], 3)
        d = b.sum(axis=1)
        return d

    def l2_norm(self,input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output
    def cos_loss(self,adv,true,target):
        v1 = self.l2_norm(adv)  # 对抗样本向量
        v2_1 = self.l2_norm(true).detach_()  # 原图向量
        v2_2 = self.l2_norm(target).detach_()  # 目标图片向量
        tmp1 = (v1 * v2_1).sum(axis=1)  # 对抗样本 与 原图 的向量的内积
        tmp2 = (v1 * v2_2).sum(axis=1) # 对抗样本 与 目标图片 的向量的内积
        loss = tmp1 - tmp2
        # score1=tmp1.item()
        # score2=tmp2.item()
        return loss
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


    def input_diversity(self, image, low=100, high=112):
        """将图片格式从low随机padding到high"""
        if random.random() > self.div_prob:
            return image
        rnd = random.randint(low, high)#取l和h之间的整数
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')#上下采样进行格式转换，双线性插值法
        h_rem = high - rnd
        w_rem = high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)#两边补充黑色区域
        return padded
    def input_diversity2(self, image, low=200, high=224):
        """将图片格式从low随机padding到high"""
        if random.random() > self.div_prob:
            return image
        rnd = random.randint(low, high)#取l和h之间的整数
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')#上下采样进行格式转换，双线性插值法
        h_rem = high - rnd
        w_rem = high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)#两边补充黑色区域
        return padded

    def attack(self,
               model: nn.Module,
               inputs: torch.Tensor,
               inputs_tar:torch.Tensor,
               mask:torch.Tensor,
               labels_true: torch.Tensor,
               labels_target: torch.Tensor
               ) -> torch.Tensor:

        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)

        # setup optimizer
        # optimizer=optim.Adam([delta],lr=0.01)
        optimizer = optim.SGD([delta], lr=0.5, momentum=0.9)

        # for choosing best results
        best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=self.device)
        best_delta = torch.zeros_like(inputs)

        gaussian_smoothing = self.gaussian_kernel(self.device, kernel_size=5, sigma=1, channels=3)

        for _ in range(self.steps):
            # inputs=gaussian_smoothing(inputs)


            adv = inputs + delta
            div_adv = self.input_diversity(adv)  # 随机padding
            logits_true = model(inputs)
            logits_target = model(inputs_tar)

            logits = model(div_adv)

            # ce_loss_true = F.cross_entropy(logits, labels_true.long(), reduction='none')#与正确类之间的交叉熵
            # ce_loss_target = F.cross_entropy(logits, labels_target.long(), reduction='none')#与目标类之间的交叉熵
            # lossce = self.loss_amp * ce_loss_target - ce_loss_true

            loss11 = F.mse_loss(logits, logits_target, reduction='none').sum(axis=1)
            loss12 = F.mse_loss(logits, logits_true, reduction='none').sum(axis=1)
            loss1 = 4 * loss11 - loss12

            # loss1=self.cos_loss(logits,logits_true,logits_target)
            # loss2=F.l2
            # loss2 = ((adv - inputs) ** 2).sum(axis=1).sum(axis=1).sum(axis=1).sqrt()  # L2 loss
            loss3 = self.tv_loss((adv - inputs))  # TV loss
            # loss4 = self.inf_loss(abs(adv - inputs)) / 60

            loss = loss1+2*loss3
            # print(lossce,loss2,loss3)
            # print(loss11,loss12)
            # print(loss,loss1, 1.5*loss3)
            is_better = loss < best_loss

            best_loss[is_better] = loss[is_better]
            best_delta[is_better] = delta.data[is_better]

            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()

            # renorm gradient
            #T1:smooth the gradient 效果存疑bad
            # delta.grad = F.conv2d(delta.grad, self.TI_kernel(), stride=1,padding=1, groups=3)

            grad_norms = delta.grad.view(batch_size, -1).norm(p=float('inf'), dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            # avoid out of bound
            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)
            # delta.data *= mask
            if self.max_norm:
                delta.data.clamp_(-self.max_norm, self.max_norm)
                if self.quantize:
                    delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
        # if self.THRESHOLD:
        #     best_delta = torch.where((best_delta > -self.THRESHOLD) & (best_delta < self.THRESHOLD), torch.full_like(best_delta, 0), best_delta)

        return inputs + best_delta
        # return best_delta