# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os
from model_irse import IR_50, IR_101, IR_152
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
import matplotlib.pyplot as plt


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img_root, backbone, model_root,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):
    # pre-requisites
    assert (os.path.exists(img_root))
    print('Testing Data Root:', img_root)
    assert (os.path.exists(model_root))
    print('Backbone Model Root:', model_root)

    # load image
    img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a = int((128 - 112) / 2)  # x start
    b = int((128 - 112) / 2 + 112)  # x end
    c = int((128 - 112) / 2)  # y start
    d = int((128 - 112) / 2 + 112)  # y end
    ccropped = resized[a:b, c:d]  # center crop the image
    ccropped = ccropped[..., ::-1]  # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype=np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)

    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval()  # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())

    #     np.save("features.npy", features)
    #     features = np.load("features.npy")

    return features
import pandas as pd
import torch.nn.functional as F
def compare():
    backbone= IR_50([112, 112])
    model_root ='../../models/Backbone_IR_50_LFW_ADV_TRAIN.pth'
    #图片路径(1是原始图片，2是对抗图片)
    labels = pd.read_csv('test.csv')
    list1=[[0] * 2 for _ in range(1995)]
    a=np.zeros((1995,1,512))
    for i in range(1995):
        picname=labels.at[i, 'ImageId']
        oriname=picname.split('__')[1]
        advname=os.path.splitext(oriname)[0]+'.jpg'
        img1 = os.path.join('../../data/single_dir/images/' +labels.at[i, 'ImageId'])
        # img2 = os.path.join('../../advSamples_images/images145.47/' + str(labels.at[idx,'TrueLabel']) + '/' + advname)
        #提取特征
        # model(img1)
        emb1 = extract_feature(img1,backbone,model_root)
        a[i]=emb1
    print(a[5])
    for i in range(1995):
        best = 0
        bestnum = 0
        for k in range(i,1995):
            diff = F.mse_loss(a[i], a[k]).sum()
            if diff > best:
                best=diff
                bestnum=k
        list1[i][0]=best*100
        list1[i][1]=labels.at[bestnum, 'ImageId']
    clm=['maxloss','bestID']
    dataframe = pd.DataFrame(lists1, columns=clm)
    dataframe.to_csv("target.csv", encoding="utf_8_sig")
    # for k in range(i,1995):
    #     img2=os.path.join('../../data/single_dir/images/' +labels.at[k, 'ImageId'])
    #     emb2 = extract_feature(img2, backbone, model_root)
    #     diff =F.mse_loss(emb1, emb2).sum()
    #
    # print(best,bestnum)

compare()
