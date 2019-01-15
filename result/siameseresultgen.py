from keras.models import *
import numpy as np
import pandas as pd
import cv2
import os
def fetch_test_imglist(DIR,IMG_ROW,IMG_COL):
    imglist=[]
    for path in os.listdir(DIR):
        imgpath=DIR+path
        img=cv2.imread(imgpath)
        img=cv2.resize(img,(IMG_ROW,IMG_COL))
        imglist.append(img)
    return np.asarray(imglist)

def fetch_train_imglist(DIR,csvpath,IMG_ROW,IMG_COL):
    traindata=pd.read_csv(csvpath)
    pathlist=traindata['Image']
    labellist=traindata['Id']
    imglist=[]
    for path in pathlist:
        img=cv2.imread(DIR+path)
        img=cv2.resize(img,(IMG_ROW,IMG_COL))
        imglist.append(img)
    return np.asarray(imglist),labellist

def duplicate_test_img(img,train_imglist):
    imglist=[]
    for _ in range(len(train_imglist)):
        imglist.append(img)
    return np.asarray(imglist)

def similiarityarr(test_img,train_imglist,train_labellist,modelpath):
    model=load_model(modelpath)
    test_imglist=duplicate_test_img(test_img,train_imglist)
    similiaritylist=model.predict([train_imglist,test_imglist])
    similiaritylist=np.reshape(similiaritylist,(len(similiaritylist)))
    rankedsimiliaritylist=sorted(similiaritylist,reverse=True)
    predictlabellist=[]
    for i,similiarity in enumerate(rankedsimiliaritylist):
        index1=list(similiaritylist).index(similiarity)
        predictlabel=train_labellist[index1]
        if(predictlabel not in predictlabellist):
            predictlabellist.append(predictlabel)
        if(len(predictlabellist)==5):
            break
    return ','.join(predictlabellist)

traindata=pd.read_csv('../train.csv')
print(len(traindata))

