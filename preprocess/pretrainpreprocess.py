import pandas as pd
import numpy as np
from imggen.fetchimg import imgarr
from keras.preprocessing.image import ImageDataGenerator

import cv2
def label_num(traindata,num_thresold):
    labelnum=traindata.groupby('Id').size()
    labelnum1=labelnum[labelnum>num_thresold]
    return labelnum1

def preprocess(traindata,num_thresold):
    labelnum=label_num(traindata,num_thresold)
    #print(len(labelnum))
    indexlist=[]
    for i,label in enumerate(traindata['Id']):
        if(label in labelnum.index and label!='new_whale'):
            indexlist.append(traindata.index[i])
    return traindata['Image'][indexlist],traindata['Id'][indexlist],len(labelnum)

def img_all_list(BASE_DIR,IMG_ROW,IMG_COL,imgpathlist):
    imglist=[]
    for path in imgpathlist:
        imgpath=BASE_DIR+path
        img=imgarr(imgpath)
        img=cv2.resize(img,(IMG_ROW,IMG_COL))
        imglist.append(img)
    return np.asarray(imglist)

def imggen(imglist,labellist,gen_times=20):
    imggenlist=[]
    labelgenlist=[]
    datagen=ImageDataGenerator(
        width_shift_range=0.25,height_shift_range=0.25,
        brightness_range=[0.1,4],
    )
    datagen.fit(imglist)
    for i,(batch_data,batch_label) in enumerate(datagen.flow(imglist,labellist)):

        for j in range(len(batch_data)):
            imggenlist.append(batch_data[j])
            labelgenlist.append(batch_label[j])
        if(i>=gen_times):
            break
    imggenlist=np.asarray(imggenlist)
    labelgenlist=np.asarray(labelgenlist)
    imglist=np.concatenate((imglist,imggenlist),axis=0)
    labellist=np.concatenate((labellist,labelgenlist),axis=0)
    return imglist,labellist

