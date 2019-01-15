import numpy as np
import random
import cv2
from imggen.fetchimg import imgarr,siamese_fetch_img
import os
def triplet_train_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,batch_size=50):
    while(True):
        anchorlist=[]
        positivelist=[]
        negativelist=[]
        for i in range(batch_size):
            rndid=random.randint(0,len(pathdata)-1)
            imgpath=BASE_DIR+pathdata[pathdata.index[rndid]]
            anchor=imgarr(imgpath)
            anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
            anchorlist.append(anchor)
            positive=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                            labeldata,pathdata,kind=1)
            negative=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                   labeldata,pathdata,kind=0)
            positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
            negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
            positivelist.append(positive)
            negativelist.append(negative)
        yield ([np.asarray(anchorlist),np.asarray(positivelist),np.asarray(negativelist)],
               None)

def triplet_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata):
    anchorlist=[]
    positivelist=[]
    negativelist=[]
    for i in range(len(pathdata)):
        imgpath=BASE_DIR+pathdata[pathdata.index[i]]
        print(imgpath)
        print(os.path.exists(imgpath))
        anchor=imgarr(imgpath)
        anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
        positive=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[i]],
                               labeldata,pathdata,kind=1)
        negative=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[i]],
                                   labeldata,pathdata,kind=0)
        positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
        negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
        anchorlist.append(anchor)
        positivelist.append(positive)
        negativelist.append(negative)
    return ([np.asarray(anchorlist),np.asarray(positivelist),np.asarray(negativelist)],
            None)

def triplet_train_gen1(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata,batch_size=50):
    while(True):
        imglist=[]
        for i in range(batch_size):
            rndid=random.randint(0,len(pathdata)-1)
            imgpath=BASE_DIR+pathdata[pathdata.index[rndid]]
            anchor=imgarr(imgpath)
            anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
            positive=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                       labeldata,pathdata,kind=1)
            negative=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[rndid]],
                                       labeldata,pathdata,kind=0)
            positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
            negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
            imglist.append([anchor,positive,negative])
        imglist=np.asarray(imglist)
        yield ([imglist[:,0],imglist[:,1],imglist[:,2]])

def triplet_valid_gen1(BASE_DIR,IMG_ROW,IMG_COL,pathdata,labeldata):
    imglist=[]
    for i in range(len(pathdata)):
        imgpath=BASE_DIR+pathdata[pathdata.index[i]]
        anchor=imgarr(imgpath)
        anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
        positive=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[i]],
                                   labeldata,pathdata,kind=1)
        negative=siamese_fetch_img(BASE_DIR,labeldata[labeldata.index[i]],
                                   labeldata,pathdata,kind=0)
        positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
        negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
        imglist.append([anchor,positive,negative])
    imglist=np.asarray(imglist)
    return ([imglist[:,0],imglist[:,1],imglist[:,2]])