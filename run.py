from torch import nn
import timm
import scipy.stats as st
from torch.nn import DataParallel
import torch.nn.functional as F
import random
import argparse
import os
import numpy as np
import torch
# from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from torch.autograd import Variable as V
from typing import Optional
from guass import get_gaussian_blur

class Attacker1:
    def __init__(self,
                 steps: int,
                 quantize: bool = True,#取整
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 THRESHOLD:Optional[float] = None,
                 amplification:float=1.5,
                 div_prob: float = 0.9,
                 loss_amp: float = 4.0,
                 device: torch.device = torch.device('cuda:0')) -> None:
        self.steps = steps

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.div_prob = div_prob
        self.loss_amp = loss_amp
        self.amplification=amplification
        self.THRESHOLD=THRESHOLD

        self.device = device

    def torch_staircase_sign(self,noise, n):
        noise_staircase = torch.zeros(size=noise.shape).cuda()
        sign = torch.sign(noise).cuda()
        temp_noise = noise.cuda()
        abs_noise = abs(noise)
        base = n / 100
        percentile = []
        for i in np.arange(n, 100.1, n):
            percentile.append(i / 100.0)
        medium_now = torch.quantile(abs_noise.reshape(len(abs_noise), -1),
                                    q=torch.tensor(percentile, dtype=torch.float32).cuda(), dim=1,
                                    keepdim=True).unsqueeze(2).unsqueeze(3)

        for j in range(len(medium_now)):
            # print(temp_noise.shape)
            # print(medium_now[j].shape)
            update = sign * (abs(temp_noise) <= medium_now[j]) * (base + 2 * base * j)
            noise_staircase += update
            temp_noise += update * 1e5

        return noise_staircase

    def ensemble_input_diversity(self,input_tensor, idx):
        # [560,620,680,740,800] --> [112, 120, 140, 180]
        image_width=112
        if random.random()<0.3:
            return input_tensor
        else:
            list=[120, 140]
            rnd = torch.randint(image_width, list[idx], ())
            rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
            h_rem = list[idx] - rnd
            w_rem = list[idx] - rnd
            pad_top = torch.randint(0, h_rem, ())
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_rem, ())
            pad_right = w_rem - pad_left
            pad_list = (pad_left, pad_right, pad_top, pad_bottom)
            padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
            padded = nn.functional.interpolate(padded, [112, 112], mode='bilinear')
            return padded

    def gkern(self,kernlen, nsig):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    def get_gaussian_kernel(self):
        kernel_size = 3
        kernel = self.gkern(kernel_size, 2).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
        return gaussian_kernel


    def project_kern(self,kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).cuda()
        return stack_kern, kern_size // 2

    def project_noise(self,x, stack_kern, kern_size):
        # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
        x = F.conv2d(x, stack_kern, padding = (kern_size, kern_size), groups=3)
        return x
    def get_stack_kern(self):
        stack_kern, kern_size = self.project_kern(3)
        return stack_kern, kern_size

    def clip_by_tensor(self,t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def modelcompute(self,adv,pre_grad,model,idx):#计算advlogits
        output = 0
        output+=model(F.interpolate(self.ensemble_input_diversity(adv + pre_grad, idx), (112, 112), mode='bilinear'))
        return output
    def modelcompute1(self,input,model):#计算logits
        output = 0
        output+=model(input)
        return output

    def attack(self,
               models: nn.Module,
               inputs: torch.Tensor,
               inputs_tar:torch.Tensor,
               mask:torch.Tensor,
               labels_true: torch.Tensor,
               labels_target: torch.Tensor
               ) -> torch.Tensor:
        x=inputs
        max_epsilon=self.max_norm
        images_min = self.clip_by_tensor(x - max_epsilon / 255.0, 0.0, 1.0)
        images_max = self.clip_by_tensor(x + max_epsilon / 255.0, 0.0, 1.0)
        x_min, x_max=images_min,images_max
        eps = self.max_norm / 255.0
        num_iter = self.steps
        alpha = eps / num_iter
        amplification=1.5
        alpha_beta = alpha * amplification
        gamma = alpha_beta
        models.zero_grad()


        # x.requires_grad = True
        adv = x.clone()
        adv = adv.cuda()
        adv.requires_grad = True
        amplification = 0.0
        pre_grad = torch.zeros(adv.shape).cuda()
        gaussian_kernel=self.get_gaussian_kernel()
        stack_kern, kern_size=self.get_stack_kern()
        best_loss = 1e4 * torch.ones(x.shape[0], dtype=torch.float, device=self.device)
        best_adv = torch.zeros_like(x)
        gaussian_blur = get_gaussian_blur(kernel_size=3, device=self.device)

        for i in range(num_iter):
            if i == 0:
                # adv = gaussian_blur(adv)
                adv = self.clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad = True)
            logits_target =self.modelcompute1(inputs_tar, models)
            logits_true = self.modelcompute1(inputs, models)

            output1=self.modelcompute(adv, pre_grad, models,0)
            loss11 = F.mse_loss(output1, logits_target, reduction='none').sum(axis=1)
            loss12 = F.mse_loss(output1, logits_true, reduction='none').sum(axis=1)
            loss1=1.5*loss11-loss12

            output2 = self.modelcompute(adv, pre_grad, models, 1)
            loss21 = F.mse_loss(output2, logits_target, reduction='none').sum(axis=1)
            loss22 = F.mse_loss(output2, logits_true, reduction='none').sum(axis=1)
            loss2 =1.5*loss21-loss22



            loss = (loss1 + loss2 ) /2.0

            loss.mean().backward()
            noise = adv.grad.data
            pre_grad = adv.grad.data
            noise = gaussian_blur(noise)

            # MI-FGSM
            # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
            # noise = momentum * grad + noise
            # grad = noise

            # PI-FGSM
            amplification += alpha_beta * self.torch_staircase_sign(noise, 1.5625)
            cut_noise =self.clip_by_tensor(abs(amplification) - eps, 0.0, 10000.0) * torch.sign(amplification)
            projection = gamma * self.torch_staircase_sign(self.project_noise(cut_noise, stack_kern, kern_size), 1.5625)

            # staircase sign method (under review) can effectively boost the transferability of adversarial examples, and we will release our paper soon.
            pert = (alpha_beta * self.torch_staircase_sign(noise, 1.5625) + 0.5 * projection)
            adv = adv + pert * mask
            # adv = adv + pert
            # print(mask.max())
            # print(mask.min())
            # exit()
            # adv = adv + alpha * torch_staircase_sign(noise, 1.5625)
            adv = self.clip_by_tensor(adv, x_min, x_max)
            adv = V(adv, requires_grad = True)

            is_better = loss < best_loss
            best_loss[is_better] = loss[is_better]
            best_adv[is_better] = adv[is_better]

        return best_adv.detach()


    def main1(self):
        # adv_img = self.graph(images, gt, images_min, images_max, model)
        pass

    #     with torch.no_grad():
    #         sum_dense += (dense(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         # sum_res += (res(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         # sum_wide_res += (wide_res(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_res += (res(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_res101 += (res101(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_vgg += (vgg(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_dense169 += (dense169(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_eff += (eff(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #         sum_wide += (wide(F.interpolate(adv_img, (256, 256))).argmax(1) != gt).detach().sum().cpu()
    #
    #         if iter % 20 == 0:
    #             batch_size = len(adv_img)
    #             print('dense = {:.2%}'.format(sum_dense / (batch_size * iter)))
    #             print('res = {:.2%}'.format(sum_res / (batch_size * iter)))
    #             print('res101 = {:.2%}'.format(sum_res101 / (batch_size * iter)))
    #             print('wide = {:.2%}'.format(sum_wide / (batch_size * iter)))
    #             print('vgg = {:.2%}'.format(sum_vgg / (batch_size * iter)))
    #             print('dense169 = {:.2%}'.format(sum_dense169 / (batch_size * iter)))
    #             print('sum_eff = {:.2%}'.format(sum_eff / (batch_size * iter)))
    #
    # # print('dense = {:.2%}'.format(sum_dense / 5000.0))
    # # print('res = {:.2%}'.format(sum_res / 5000.0))
    # # print('wide_res = {:.2%}'.format(sum_wide_res / 5000.0))
    # # print('next = {:.2%}'.format(sum_next / 5000.0))
    # # print('vgg = {:.2%}'.format(sum_vgg / 5000.0))
    # # print('xception = {:.2%}'.format(sum_xception / 5000.0))
    # # print('sum_adv = {:.2%}'.format(sum_adv / 5000.0))
    # print('res = {:.2%}'.format(sum_res / 5000))
    # print('res101 = {:.2%}'.format(sum_res101 / 5000))
    # print('wide = {:.2%}'.format(sum_wide / 5000))
    # print('dense169 = {:.2%}'.format(sum_dense169 / 5000))
    # print('vgg = {:.2%}'.format(sum_vgg / 5000))
    # print('sum_eff = {:.2%}'.format(sum_eff / 5000))

    # score_fid = cal_fid(opt.input_dir, opt.output_dir)
    # score_lpips = cal_lpips(opt.input_dir, opt.output_dir)
    # final_score = 100 * score_fid * score_lpips
    # print('score_lpips: ', score_lpips)
    # print('score_fid:', score_fid)
    # print("final score: score_ASR * ", final_score)
