import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
def kind_list(imgdata):
    kindlist=imgdata.groupby('Id').size()
    return kindlist.index

def kind_list1(imgdata):
    kindlist=imgdata.groupby('Id').size()
    kindlist1=kindlist[kindlist>1]
    return kindlist1.index

def fetch_all_kind_list(imgdata):
    kindlist=kind_list(imgdata)
    kindimgpathlist=[]
    for kind in kindlist:
        kindimgpathlist.append(list(imgdata['Image'][imgdata['Id']==kind]))
    return kindimgpathlist,kindlist

def fetch_largest_img_index(BASE_DIR,pathlist):
    sizelist=[]
    for path in pathlist:
        imgpath=BASE_DIR+path
        img=cv2.imread(imgpath)
        sizelist.append(img.shape[0]*img.shape[1])
    rankedsizelist=sorted(sizelist,reverse=True)
    return list(sizelist).index(rankedsizelist[0])

def fetch_all_kind_img_list(BASE_DIR,IMG_ROW,IMG_COL,imgdata):
    kindlist=kind_list(imgdata)
    kindimgpathlist=[]
    imglist=[]
    for kind in kindlist:
        pathlist=list(imgdata['Image'][imgdata['Id']==kind])
        kindimgpathlist.append(pathlist)
        index1=fetch_largest_img_index(BASE_DIR,pathlist)
        imgpath=BASE_DIR+pathlist[index1]
        img=cv2.imread(imgpath)
        img=cv2.resize(img,(IMG_ROW,IMG_COL))
        imglist.append(img)
    return kindimgpathlist,kindlist,imglist


def fetch_all_kind_img_bound_list(BASE_DIR,IMG_ROW,IMG_COL,joinimgdata):
    kindlist=kind_list(joinimgdata)
    imgalllist=[]
    trainimgalllist=[]
    validimgalllist=[]
    for kind in kindlist:
        pathlist=list(joinimgdata['Image'][joinimgdata['Id']==kind])
        x0list=list(joinimgdata['x0'][joinimgdata['Id']==kind])
        y0list=list(joinimgdata['y0'][joinimgdata['Id']==kind])
        x1list=list(joinimgdata['x1'][joinimgdata['Id']==kind])
        y1list=list(joinimgdata['y1'][joinimgdata['Id']==kind])
        imglist=[]
        for (path,x0,y0,x1,y1) in zip(pathlist,x0list,y0list,x1list,y1list):
            img=cv2.imread(BASE_DIR+path)
            img1=img[x0:x1,y0:y1,:]
            if(img1.shape[0]==0 or img1.shape[1]==0):
                img1=img
            img1=cv2.resize(img1,(IMG_ROW,IMG_COL))
            imglist.append(img1)
        if(len(imglist)<3):
            trainimgalllist.append(imglist)
            validimgalllist.append(imglist)
        else:
            trainimgalllist.append(imglist[:int(len(imglist)*0.8)])
            validimgalllist.append(imglist[int(len(imglist)*0.8):])
        imgalllist.append(np.asarray(imglist))
    return kindlist,trainimgalllist,validimgalllist

def fetch_all_kind_list1(imgdata):
    kindlist=kind_list1(imgdata)
    kindimgpathlist=[]
    for kind in kindlist:
        kindimgpathlist.append(list(imgdata['Image'][imgdata['Id']==kind]))
    return kindimgpathlist,kindlist

def fetch_kind_list_split(kindimgpathlist,split_size=0.8):
    trainkindimgpathlist=[]
    validkindimgpathlist=[]
    for pathlist in kindimgpathlist:
        if(len(pathlist)<=3):
            trainkindimgpathlist.append(pathlist)
            validkindimgpathlist.append(pathlist)
        else:
            trainkindimgpathlist.append(pathlist[:int(len(pathlist)*split_size)])
            validkindimgpathlist.append(pathlist[int(len(pathlist)*split_size):])
    return trainkindimgpathlist,validkindimgpathlist