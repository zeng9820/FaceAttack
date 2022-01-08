import os
import random
import torch
import numpy as np
import glob
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

class ImageNet_A(Dataset):
    def __init__(self, root_dir='../../data', csv_name='test5.csv', folder_name='images',transform=to_torch_tensor):
        #testmin用的是扩充数据集作为target，testori用的是最相近的图像作为target，testadv用的是最不相近的图像作为target用于对抗训练
        # labels_dir = os.path.join(root_dir, csv_name)
        self.image_dir = os.path.join(root_dir, folder_name)
        self.labels = pd.read_csv(csv_name)
        self.transform=to_torch_tensor

    def __len__(self):
        l = len(self.labels)
        return l

    def __getitem__(self, idx):
        # filename = os.path.join('../../data/single_dir1/images', self.labels.at[idx, 'ImageId'])#原始图片名字#single_dir1原始大小
        filename = os.path.join('../../data/single_dir_2',self.labels.at[idx, 'ImageId'])
        # tarname=os.path.join('../../data/single_pic/images1',self.labels.at[idx,'TargetId'])#
        tarname = os.path.join('../../data/single', self.labels.at[idx, 'TargetId'])
        # maskname=os.path.join('../../data/mask/masks/masks', self.labels.at[idx, 'ImageId'])
        # mask1=cv2.imread(maskname)
        # mask1 = mask1[:, :, ::-1]  # 转换成RGB格式
        # mask11 = np.transpose(mask1.astype(np.float32),axes=[2, 0, 1])  # [width,height,channels]转换成[channels,width,height]
        # mask = mask11 / 255.0
        # mask=1

        in_img_t = cv2.imread(filename)
        in_img_t = cv2.resize(in_img_t, (112, 112), interpolation=cv2.INTER_NEAREST)#resize224
        in_img_t=in_img_t[:, :, ::-1]#转换成RGB格式
        in_img = np.transpose(in_img_t.astype(np.float32), axes=[2, 0, 1])#[width,height,channels]转换成[channels,width,height]
        img = in_img / 255.0

        in_img_t_tar = cv2.imread(tarname)[:, :, ::-1]  # 转换成RGB格式#改
        in_img_t_tar = cv2.resize(in_img_t_tar,(112, 112), interpolation=cv2.INTER_NEAREST)  # resize224
        in_img_tar = np.transpose(in_img_t_tar.astype(np.float32),
                                  axes=[2, 0, 1])  # [width,height,channels]转换成[channels,width,height]
        img_tar = in_img_tar / 255.0


        label_true =self.labels.at[idx, 'TrueLabel']
        label_true=torch.tensor(label_true)
        # label_true = F.one_hot(torch.tensor(label_true),num_classes=425)#转换成one-hot表示
        label_target =self.labels.at[idx, 'TargetLabel']
        label_target = torch.tensor(label_target)
        # label_target=F.one_hot(torch.tensor(label_target), num_classes=425)  # 转换成one-hot表示
        size=self.labels.at[idx, 'size']

        return img,img_tar,label_true, label_target,filename,size