import pandas as pd
import os
import json
import numpy as np
import cv2
def write():#保存所有图像的信息
    piclist=os.listdir('../../data/single_dir/images/')
    lists = [[0] * 4 for _ in range(len(piclist))]

    piclist.sort()
    print(len(piclist))
    likelihood = json.load(open("../../likelihood_min2.json"))
    for i in range(len(piclist)):
        lists[i][0]=piclist[i]
        lists[i][1]=(piclist[i].split("__")[0])
        name_index = lists[i][1] + '.png'
        # if lists[i][1]=='120':
        #     lists[i][2] = '310'
        # elif lists[i][1]=='293':
        #     lists[i][2] = '120'
        if likelihood[name_index][0][:-4] == lists[i][1]:
            lists[i][2]=likelihood[name_index][2][:-4]
        else :
            lists[i][2] = likelihood[name_index][0][:-4]
        if likelihood[name_index][2][:-4]==likelihood[name_index][0][:-4]==likelihood[name_index][4][:-4]:
            print(lists[i][1])

        lists[i][3]=lists[i][2]+'.png'

    clm=['ImageId','TrueLabel','TargetLabel','TargetId']

    dataframe = pd.DataFrame(lists,columns=clm)
    dataframe.to_csv("testmin.csv",encoding="utf_8_sig")
    # idx=0
    labels=pd.read_csv('testmin.csv')#test.csv包含全部图片的信息,testmin是用扩充数据集作为target
    filename = os.path.join('../../data/single_dir2/images', labels.at[8, 'ImageId'])
    a=cv2.imread(filename)
    # print(a)
    print(a.shape)


def write2():  # 保存所有图像的信息
    piclist = os.listdir('../../data/single_dir_2')
    # lists = [[0] * 5 for _ in range(826)]
    lists = [[0] * 5 for _ in range(509)]

    piclist.sort()
    print(piclist[0])
    likelihood = json.load(open("../../likelihood_min3.json"))
    n1,n2,n3=0,0,0
    num=0
    for i in range(len(piclist)):
        filename = piclist[i]
        a = cv2.imread('../../data/single_dir_2/'+filename)
        size=a.shape[0]
        if size<200:
        # if size >= 180:
            n3+=1
            continue
        if size>=500:
            n1+=1
        if size<500&size>=200:
            n2+=1
        lists[num][0] = piclist[i]
        lists[num][1] = (piclist[i].split("_")[0])
        name_index = lists[num][1] + '.png'
        if likelihood[name_index][0][:-4]!=lists[num][1]:
            lists[num][2] = likelihood[name_index][0][:-4]
        else: lists[num][2] = likelihood[name_index][2][:-4]
        lists[num][3] = lists[num][2] + '.png'
        lists[num][4]=size
        num += 1
    clm = ['ImageId', 'TrueLabel', 'TargetLabel', 'TargetId','size']

    dataframe = pd.DataFrame(lists, columns=clm)
    dataframe.to_csv("test6.csv", encoding="utf_8_sig")
    #test5保存了size小于200的图片
    #test6保存了size大于等于200的图片
    print('大于500',n1)
    print('500-200',n2)
write2()
