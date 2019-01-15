import numpy as np
import pandas as pd
import cv2
from imggen.fetchimg import imgarr
import random

def extract_imgarr(BASE_DIR,pathlist,IMG_ROW,IMG_COL):
    pathpair=np.random.randint(0,len(pathlist),2)
    imglist=[]
    for i in range(2):
        path=BASE_DIR+pathlist[pathpair[i]]
        img=imgarr(path)
        img=cv2.resize(img,(IMG_ROW,IMG_COL))
        imglist.append(img)
    return imglist

def triplet_img_gen(BASE_DIR,IMG_ROW,IMG_COL,kindimgpathlist,batch_size=50,contrast_times=10):
    while(True):
        anchorlist=[]
        positivelist=[]
        negativelist=[]
        for i in range(batch_size):
            indexpair=np.random.randint(0,len(kindimgpathlist),contrast_times)
            for positiveindex in indexpair:
                negativeindex=random.choice([num for num in range(len(kindimgpathlist)) if num not in [positiveindex]])
                positiveimglist=extract_imgarr(BASE_DIR,kindimgpathlist[positiveindex],IMG_ROW,IMG_COL)
                anchor=positiveimglist[0]
                positive=positiveimglist[1]
                negative=extract_imgarr(BASE_DIR,kindimgpathlist[negativeindex],IMG_ROW,IMG_COL)[0]
                anchorlist.append(anchor)
                positivelist.append(positive)
                negativelist.append(negative)
        yield ([np.asarray(anchorlist),np.asarray(positivelist),np.asarray(negativelist)],
               None)
