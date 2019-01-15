import cv2
import os
import numpy as np
import random
import pandas as pd
from imggen.fetchimg import imgarr
def pretrain_train_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,
                       batch_size=50):
    while(True):
        imglist=[]
        labellist=[]
        for i in range(batch_size):
            rnd_id=random.randint(0,len(pathdata)-1)
            imgpath=BASE_DIR+pathdata[pathdata.index[rnd_id]]
            img=imgarr(imgpath)
            img=cv2.resize(img,(IMG_ROW,IMG_COL))
            imglist.append(img)
            if(labeldata[labeldata.index[rnd_id]]=='new_whale'):
                labellist.append(0)
            else:
                labellist.append(1)
        yield (np.asarray(imglist),
               np.asarray(labellist))

def pretrain_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata):
    imglist=[]
    labellist=[]
    for i in range(len(pathdata)):
        imgpath=BASE_DIR+pathdata[pathdata.index[i]]
        img=imgarr(imgpath)
        img=cv2.resize(img,(IMG_ROW,IMG_COL))
        imglist.append(img)
        if(labeldata[labeldata.index[i]]=='new_whale'):
            labellist.append(0)
        else:
            labellist.append(1)
    return (np.asarray(imglist),
            np.asarray(labellist))
