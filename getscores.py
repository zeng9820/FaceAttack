import imageio
import numpy as np
import torch
import pytorch_msssim
import csv
import os
import cv2
import json
import torch
import torchvision
from torch import nn
from model_irse import IR_50, IR_101, IR_152
from simple_attack import Normalize,Permute
import argparse
from datetime import datetime as dt
from loader import ImageNet_A

# 图像质量评分score_2，范围[0, 1]，越高越好
def getscore2(ori_img, adv_img):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    m = pytorch_msssim.MSSSIM()
    img1 = torch.from_numpy(ori_img.transpose((2, 0, 1)) / 255).float().unsqueeze(0)
    img2 = torch.from_numpy(adv_img.transpose((2, 0, 1)) / 255).float().unsqueeze(0)
    score2 = m(img1, img2).item()
    return score2


# 扰动大小评分规则score_1，范围[0, 1]，越高越好
def getscore1(ori_img, adv_img):
    ori_img = ori_img.astype(int)  # 图像数组，（height, weight, channels）
    adv_img = adv_img.astype(int)
    b=(adv_img - ori_img)

    # print(b[:, :, 0].max(),b[:, :, 1].max(),b[:, :, 2].max())
    dif = np.clip((adv_img - ori_img), -20, 20)  # 扰动限制在[-20, 20]的区间范围内
    # dif = adv_img - ori_img
    # score1 = 1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    a=0
    a= (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max())
    # print(a)
    score1 = 1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    return score1
import pandas as pd

class Ensemble(nn.Module):
    def __init__(self, p1,model1,p2, model2,p3, model3,p4, model4,p5, model5):#改这里
    # def __init__(self, p1, model1, p2, model2, p3, model3, p4, model4,):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5#改这里
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.p4 = p4
        self.p5 = p5#改这里

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        logits3 = self.model3(x)
        logits4 = self.model4(x)
        logits5 = self.model5(x)#改这里
        sum=self.p1+self.p2+self.p3+self.p4+self.p5
        logits_e = (logits4*self.p4 + logits5*self.p5) / 2
        return logits_e
def load_model_all():
    device=torch.device("cuda:0")
    fc = nn.Linear(512, 425)
    f1 = fc.eval().to(device)
    m1 = IR_101([112, 112]).eval().to(device)
    m2 = IR_152([112, 112]).eval().to(device)
    m3 = IR_50([112, 112]).eval().to(device)
    m4 = IR_50([112, 112]).eval().to(device)
    m5 = IR_50([112, 112]).eval().to(device)
    model1 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m1,
        f1
    )
    model2 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m2,
        f1
    )
    model3 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m3,
        f1
    )
    model4 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m4,
        f1
    )
    model5 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m5,
        f1
    )
    model1.load_state_dict(torch.load('Backbone_IR_101_Batch_108320_fc_at.pth', map_location=device))  # m1
    model2.load_state_dict(torch.load('Backbone_IR_152_MS1M_Epoch_112_fc_at.pth', map_location=device))#m2
    model3.load_state_dict(torch.load('Backbone_IR_50_LFW_ADV_TRAIN_fc_at.pth', map_location=device))  # m3
    model4.load_state_dict(torch.load('backbone_ir50_ms1m_epoch120_fc_at.pth', map_location=device))  # m4
    model5.load_state_dict(torch.load('Backbone_IR_50_LFW_fc_at.pth', map_location=device))#m5
    model1.eval().to(device)
    model2.eval().to(device)
    model3.eval().to(device)
    model4.eval().to(device)
    model5.eval().to(device)
    p1,p2,p3,p4,p5=1,1,1,1,1
    return p1,model1,p2, model2,p3, model3,p4,model4,p5,model5
def load_model():
    device=torch.device("cuda:0")
    # m1 = IR_101([112, 112]).eval().to(device)
    # m2 = IR_152([112, 112]).train().to(device)
    # m3 = IR_50([112, 112]).eval().to(device)
    # m4 = IR_50([112, 112]).eval().to(device)
    m5 = IR_50([112, 112]).eval().to(device)
    fc = nn.Linear(512, 425)
    f1=fc.eval().to(device)
    # f1.load_state_dict(torch.load('Backbone_IR_101_Batch_108320_fc.pth', map_location=device))
    model1 = nn.Sequential(
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        Permute([2, 1, 0]),
        m5,
        f1
    )
    # model1.load_state_dict(torch.load('Backbone_IR_101_Batch_108320_fc_at.pth', map_location=device))#m1
    # model1.load_state_dict(torch.load('Backbone_IR_152_MS1M_Epoch_112_fc_at.pth', map_location=device))#m2
    # model1.load_state_dict(torch.load('Backbone_IR_50_LFW_ADV_TRAIN_fc_at.pth', map_location=device))#m3
    # model1.load_state_dict(torch.load('backbone_ir50_ms1m_epoch120_fc_at.pth', map_location=device))#m4
    model1.load_state_dict(torch.load('Backbone_IR_50_LFW_fc_at.pth', map_location=device))#m5
    model1.eval().to(device)
    return model1
def writescores(net):
    piclist = os.listdir('../../data/single_dir/images/')
    lists = [[0] * 8 for _ in range(len(piclist))]
    piclist.sort()
    likelihood = json.load(open("../../likelihood_min.json"))
    n1, n2 = 0, 0
    for i in range(len(piclist)):
        oriname = os.path.splitext(piclist[i].split("__")[1])[0] + '.jpg'
        filename = os.path.join('../../data/demo/images/', piclist[i].split("__")[0] + '/' + oriname)
        advname=os.path.join('../../advSamples_images/images/', piclist[i].split("__")[0] + '/' + oriname)
        if not os.path.exists(advname):
            continue
        a = cv2.imread(filename)
        b=cv2.imread('../../data/single_dir2/images/'+piclist[i])
        adv=cv2.imread(advname)
        img=adv
        img = img[:, :, ::-1]  # 转换成RGB格式
        img=cv2.resize(img, (112, 112),interpolation=cv2.INTER_NEAREST)
        img = np.transpose(img.astype(np.float32),
                              axes=[2, 0, 1])  # [width,height,channels]转换成[channels,width,height]
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze_(0).to(torch.device("cuda:0"))
        pre_class=net(img).argmax(axis=1).cpu().numpy()
        size = a.shape[0]
        # if size > 500:
        #     n1 += 1
        # if size <= 500 & size > 223:
        #     n2 += 1
        lists[i][0] = piclist[i]
        lists[i][1] = (piclist[i].split("__")[0])
        name_index = lists[i][1] + '.png'
        lists[i][2] = likelihood[name_index][2][:-4]
        lists[i][3] = lists[i][2] + '.png'
        lists[i][4] = size#size
        lists[i][5]=getscore1(a,adv)
        lists[i][6]=getscore2(a,adv)
        lists[i][7]=pre_class[0]
    a=0
    n=0
    for i in range(len(piclist)):
        if lists[i][7]!=int(lists[i][1]):
            n+=1
            a+=lists[i][5]+lists[i][6]
    print(n,a)
    clm = ['ImageId', 'TrueLabel', 'TargetLabel', 'TargetId', 'size','scores1','scores2','pre_class']#,
    dataframe = pd.DataFrame(lists, columns=clm)
    dataframe.to_csv("scores.csv", encoding="utf_8_sig")
if __name__ == '__main__':
    device = torch.device("cuda:0")
    p1, model1, p2, model2, p3, model3, p4, model4, p5, model5 = load_model_all()
    model = Ensemble(p1, model1, p2, model2, p3, model3, p4, model4, p5, model5)
    model.cuda()
    model.eval()
    writescores(model)

