from keras.models import Model,load_model
import numpy as np
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

def duplicate_test_img(img,train_imglist):
    imglist=[]
    for _ in range(len(train_imglist)):
        imglist.append(img)
    return np.asarray(imglist)

def similiarityarr(BASE_DIR,IMG_ROW,IMG_COL,test_img,kindimglist,kindlist,model):
    imglist1=[]
    for i in range(len(kindimglist)):
        imglist1.append(test_img)
    labellist=[]
    similiaritylist=model.predict([np.asarray(kindimglist),np.asarray(imglist1)])
    similiaritylist=np.reshape(similiaritylist,(len(similiaritylist)))
    rankedsimiliarity=sorted(similiaritylist,reverse=True)
    for i in range(5):
        labelindex=kindlist[list(similiaritylist).index(rankedsimiliarity[i])]
        labellist.append(labelindex)
    return ','.join(labellist)



