from PIL import Image
import pandas as pd
import os
import cv2
from shutil import copy
import numpy as np
def moveori_pic():
    #将原始图片大小移到single_dir1/images中
    csv_name='test2.csv'
    labels = pd.read_csv(csv_name)
    for i in range(1995):
        filename = os.path.splitext(labels.at[i, 'ImageId'])[0].split("__")[1]
        print(filename)
        dirname=str(labels.at[i,'TrueLabel'])
        oripath=("../../dataori/01 初赛 复赛（阶段1）数据集/01 初赛+复赛（阶段1）数据集/images" + '/' + dirname + '/' + filename + '.jpg')
        copy(oripath,'../../data/single_dir1/images')
        os.renames('../../data/single_dir1/images/'+filename + '.jpg','../../data/single_dir1/images/'+labels.at[i, 'ImageId'])
def targetpic():
    #扩充目标集
    extpic='../../data/single_pic/lfw_cut'
    path='../../data/single_pic/images1/'
    list=sorted(os.listdir(extpic))
    n=0
    for i in range(len(os.listdir(extpic))):
        if not os.path.exists(path+str(i)+'.png'):
            list1=sorted(os.listdir(extpic+'/'+list[i-425]))
            if not list1:
                continue
            copy(extpic+'/'+list[i-425]+'/'+list1[0],path)
            os.renames(path+list1[0],path+str(i)+'.png')
            n+=1
    print('扩充完成数量：',n+425)
# targetpic()
def trainpic():
    #扩充训练集，进行AT对抗训练，图片存放地址为：data/single_dir2/images
    advpic='../../advSamples_images'
    list1=sorted(os.listdir(advpic))#文件夹目录
    path='../../data/single_dir2/images/'
    num=0
    for dir in list1:
        list2 = sorted(os.listdir('../../advSamples_images/'+dir))
        num+=1
        for dir2 in list2:
            list3 = sorted(os.listdir('../../advSamples_images/' + dir+'/'+dir2))
            for pic in list3:#图片名
                filename=os.path.splitext(pic)[0]
                img=cv2.imread('../../advSamples_images/' + dir+'/'+dir2+'/'+pic)
                numpy_img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_NEAREST)
                numpy_img = numpy_img[:, :, ::-1]
                Image.fromarray(np.array(numpy_img).astype('uint8')).save(
                    path + dir2+'__'+filename +'_'+str(num)+ '.png', quality=95)
                # os.rename(path + filename + '.png',
                #           path + dir2+'__'+filename +'_'+str(num)+ '.png')

trainpic()




