# Helper function for extracting features from pre-trained models
import sys, os
import argparse
from random import random
from model_irse import IR_50, IR_101, IR_152,IR_SE_50
from model_irse2 import IR_SE_100
from ResNet2 import iresnet200,iresnet100,iresnet50
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob
import torchvision
from attacker import Attacker
from run import Attacker1
from run2 import Attacker2
from loader import ImageNet_A
from torchattacks.attacks.autoattack import AutoAttack
# sys.path.append(os.path.abspath('./utils'))
# from Normalize import Normalize, Permute
from datetime import datetime as dt
from PIL import Image
class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]

        return x


class Permute(nn.Module):

    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]
# Return the torch.tensor image pool

class Ensemble(nn.Module):
    def __init__(self,model1,model2,model3):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        # self.model4 = model4
        # self.model5 = model5#改这里


    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        logits3 = self.model3(x)
        # logits4 = self.model4(x)
        # logits5 = self.model5(x)#改这里
        # sum=self.p1+self.p2+self.p3+self.p4+self.p5
        # logits_e = (logits1*self.p1 + logits2*self.p2 + logits3*self.p3+ logits4*self.p4+logits5*self.p5) / sum
        # return logits_e
        return (logits1+logits2+logits3)/3
class Ensemble1(nn.Module):
    def __init__(self, p1,model1,p2, model2,p3, model3,p4, model4,p5, model5):#改这里
        super(Ensemble1, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5#改这里
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.p4=p4
        self.p5=p5#改这里

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        logits3 = self.model3(x)
        logits4 = self.model4(x)
        logits5 = self.model5(x)#改这里
        sum=self.p1+self.p2+self.p3+self.p4+self.p5
        logits_e = (logits1*self.p1 + logits2*self.p2 + logits3*self.p3+ logits4*self.p4+logits5*self.p5) / sum
        return logits_e

def load_model():
    device=torch.device("cuda:0")

    m=IR_SE_50([224,224]).eval().to(device)
    m0 = IR_50([224, 224]).eval().to(device)
    m1 = IR_101([224, 224]).eval().to(device)

    root = '/home/liukun/ZengEn/competition/OPPO_Security/OPPO_ADVERSARIAL_ATTACK-master/attacks/M-DI2-FGSM/face.evoLVe_master/face.evoLVe_master/'
    m.load_state_dict(torch.load(root+'model/Backbone_IR_SE_50_Epoch_2_Batch_9752_Time_2021-10-12-19-39_checkpoint.pth', map_location=device))
    m0.load_state_dict(torch.load(root+'model/Backbone_IR_50_Epoch_13_Batch_63388_Time_2021-10-13-20-22_checkpoint.pth', map_location=device))
    m1.load_state_dict(torch.load(root + 'model/Backbone_IR_101_Epoch_6_Batch_39006_Time_2021-10-15-15-15_checkpoint.pth',map_location=device))
    model = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m
    )
    model0 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m0
    )
    model1 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m1
    )
    model.eval().to(device)
    model0.eval().to(device)
    model1.eval().to(device)

    return model,model0,model1
def load_model1():
    device = torch.device("cuda:0")

    m1 = IR_101([112, 112]).eval().to(device)
    m2 = IR_152([112, 112]).eval().to(device)
    m3 = IR_50([112, 112]).eval().to(device)
    m4 = IR_50([112, 112]).eval().to(device)
    m5 = IR_50([112, 112]).eval().to(device)
    # m1.load_state_dict(torch.load('../../models/Backbone_IR_101_Batch_108320.pth', map_location=device))  # m1
    m1.load_state_dict(torch.load('../../models/Backbone_IR_101_Batch_108320.pth', map_location=device))  # m1
    m2.load_state_dict(torch.load('../../models/Backbone_IR_152_MS1M_Epoch_112.pth', map_location=device))#m2
    m3.load_state_dict(torch.load('../../models/Backbone_IR_50_LFW_ADV_TRAIN.pth', map_location=device))  # m3
    m4.load_state_dict(torch.load('../../models/backbone_ir50_ms1m_epoch120.pth', map_location=device))  # m4
    m5.load_state_dict(torch.load('../../models/Backbone_IR_50_LFW.pth', map_location=device))#m5

    model1 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m1
        # f1
    )
    model2 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m2
        # f1
    )
    model3 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m3
    )
    model4 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m4
    )
    model5 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m5
    )

    model1.eval().to(device)
    model2.eval().to(device)
    model3.eval().to(device)
    model4.eval().to(device)
    model5.eval().to(device)
    p1, p2, p3, p4, p5 = 1, 1, 1, 1, 1
    return p1,model1,p2, model2,p3, model3,p4,model4,p5,model5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='.../data/', type=str, help='path to data')
    parser.add_argument('--output_dir', default='.../advSamples_images/', type=str, help='path to results')
    parser.add_argument('--batch_size', default=40, type=int, help='mini-batch size')
    parser.add_argument('--steps', default=80, type=int, help='iteration steps')
    parser.add_argument('--steps2', default=40, type=int, help='iteration steps')
    parser.add_argument('--max_norm', default=10.5, type=float, help='Linf limit')
    parser.add_argument('--THRESHOLD', default=2, type=float, help='minimal limit')
    parser.add_argument('--div_prob', default=0.9, type=float, help='probability of diversity')
    parser.add_argument('--num_class', default=425, type=int, help='number of classes')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, 'images')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # ensemble model
    p1,model1,p2, model2,p3, model3 ,p4,model4,p5,model5= load_model1()
    # model1,model2,model3=load_model()
    model = Ensemble1(p1,model1,p2, model2,p3, model3,p4,model4,p5,model5)
    # model=Ensemble(model1,model2,model3)
    model.cuda()
    model.eval()

    # set dataset
    dataset = ImageNet_A(args.input_dir)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False)

    # set attacker
    attacker = Attacker(steps=args.steps,
                        max_norm=args.max_norm / 255.0,
                        THRESHOLD=args.THRESHOLD/255.0,
                        div_prob=args.div_prob,
                        device=torch.device('cuda:0'))
    attacker1 = Attacker1(steps=args.steps2,
                        max_norm=args.max_norm,
                        THRESHOLD=args.THRESHOLD / 255.0,
                        div_prob=args.div_prob,
                        device=torch.device('cuda:0'))
    attacker2 = Attacker2(steps=args.steps2,
                          max_norm=args.max_norm/255.0,
                          THRESHOLD=args.THRESHOLD / 255.0,
                          div_prob=args.div_prob,
                          device=torch.device('cuda:0'))
    picnum=0
    curr_time = dt.now().strftime('%F %T')
    str_curr_time = dt.strptime(curr_time, "%Y-%m-%d %H:%M:%S")
    print("开始时间：",str_curr_time)

    # attack = AutoAttack(model, norm='Linf', eps=12/255, version='plus', n_classes=425, seed=None,verbose=False)
    # adv_images = attack(images, labels)

    for ind, (img,img_tar,label_true, label_target,  filenames,size) in enumerate(loader):
        print("已经完成{}个样本".format(picnum*args.batch_size))

        # run attack
        # adv = attack(img.cuda(), label_true.cuda(),label_target.cuda())
        adv = attacker2.attack(model, img.cuda(),img_tar.cuda(),label_true.cuda(), label_target.cuda())
        picnum+=1

        # save results
        for bind, filename in enumerate(filenames):
            out_img = adv[bind].detach().cpu().numpy()#生成的对抗样本，array格式
            delta_img1 = (out_img - img[bind].detach().cpu().numpy())  # 生成的对抗扰动，array格式

            delta_img = np.abs(out_img - img[bind].numpy()) * 255.0

            print('Attack on {}:'.format(os.path.split(filename)[-1]))
            print('Max: {0:.0f}, Mean: {1:.2f}'.format(np.max(delta_img),np.mean(delta_img)))

            #获取保存路径和原始样本尺寸
            pngname=os.path.split(filename)[-1]
            filenameori = os.path.splitext(pngname)[0]
            dirname = filenameori.split("_")[0]
            if os.path.exists('../../advSamples_images') is False:
                os.mkdir('../../advSamples_images' )
            if not os.path.exists('../../advSamples_images/images'):
                os.makedirs('../../advSamples_images/images')
            if not os.path.exists('../../advSamples_images/images/' + dirname):
                os.makedirs('../../advSamples_images/images/' + dirname)
            old_path = '../../data/single_dir_2'
            # #直接将对抗样本resize
            # out_img = np.transpose(out_img,axes=[1, 2, 0]) * 255.0  # 将[channels,width,height]转换成[width,height,channels]
            # out_img = out_img[:, :, ::-1]  # 转换成BGR格式
            # numpy_adv_sample = cv2.resize(out_img, (h, w), interpolation=cv2.INTER_NEAREST)
            size1 = cv2.imread(old_path + '/' + filenameori + '.jpg').shape[0]
            print(size1)
            #将对抗扰动resize再加到原始样本上
            delta_img1 = np.transpose(delta_img1,axes=[1, 2, 0]) * 255.0  # 将[channels,width,height]转换成[width,height,channels]
            numpy_adv_sample = cv2.resize(delta_img1, (size1, size1),interpolation=cv2.INTER_NEAREST)
            numpy_adv_sample = np.clip(numpy_adv_sample, -args.max_norm, args.max_norm)
            ori_pic=Image.open(old_path + '/'  + filenameori + '.jpg').convert('RGB')
            # ori_pic=cv2.imread(old_path + '/' + dirname + '/' + filename + '.jpg')
            # ori_pic =ori_pic[:, :, ::-1]
            numpy_adv_sample=(numpy_adv_sample+ori_pic).clip(0,255)
            # # print(numpy_adv_sample.shape)
            Image.fromarray(np.array(numpy_adv_sample).astype('uint8')).save(
                '../../advSamples_images/images/' + dirname + '/' + filenameori + '.png', quality=95)
            os.rename('../../advSamples_images/images/' + dirname + '/' + filenameori + '.png',
                      '../../advSamples_images/images/' + dirname + '/' + filenameori + '.jpg')
    curr_time1 = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    str_curr_time1 = dt.strptime(curr_time1, "%Y-%m-%d %H:%M:%S")
    t_time = str_curr_time1 - str_curr_time
    print("花费时间：", t_time)