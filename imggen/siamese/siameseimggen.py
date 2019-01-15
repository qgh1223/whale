import numpy as np
import random
import cv2
from imggen.fetchimg import imgarr,siamese_fetch_img

def siamese_train_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,batch_size=50):
    while(True):
        imglist1=[]
        imglist2=[]
        labellist=[]
        for i in range(batch_size):
            rndid=random.randint(0,len(pathdata)-1)
            imgpath=BASE_DIR+pathdata[pathdata.index[rndid]]
            img1=imgarr(imgpath)
            if(i%2==0):
                kind=0
                labellist.append(0)
            else:
                kind=1
                labellist.append(1)
            img2=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                           labeldata,pathdata,kind=kind)
            img1=cv2.resize(img1,(IMG_ROW,IMG_COL))

            img2=cv2.resize(img2,(IMG_ROW,IMG_COL))
            imglist1.append(img1)
            imglist2.append(img2)
        yield ([np.asarray(imglist1),np.asarray(imglist2)],
               np.asarray(labellist))

def siamese_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata):
    imglist1=[]
    imglist2=[]
    labellist=[]
    for i in range(len(pathdata)):
        imgpath=BASE_DIR+pathdata[pathdata.index[i]]
        img1=imgarr(imgpath)
        if(i%2==0):
            kind=0
            labellist.append(0)
        else:
            kind=1
            labellist.append(1)
        img2=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[i]],
                               labeldata,pathdata,kind=kind)
        img1=cv2.resize(img1,(IMG_ROW,IMG_COL))
        img2=cv2.resize(img2,(IMG_ROW,IMG_COL))
        imglist1.append(img1)
        imglist2.append(img2)
    return ([np.asarray(imglist1),np.asarray(imglist2)],
            np.asarray(labellist))
