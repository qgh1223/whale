import numpy as np
import pandas as pd
from keras.models import Model
import cv2
from imggen.triplet.tripletmining import cosine_distance,semi_hard_mining

def flatten_kindpathlist(kindpathlist):
    imgpathlist=[]
    for kindpath in kindpathlist:
        imgpathlist=np.concatenate([imgpathlist,kindpath])
    return imgpathlist

def fetch_path_imglist(BASE_DIR,IMG_ROW,IMG_COL,imgpathlist):
    imglist=[]
    for path in imgpathlist:
        imgpath=BASE_DIR+path
        img=cv2.imread(imgpath)
        img=cv2.resize(img,(IMG_ROW,IMG_COL))
        imglist.append(img)
    return np.asarray(imglist)

def triplet_mining_imggen(BASE_DIR,IMG_ROW,IMG_COL,
                          kindpathlist,n_class_per_batch,
                          n_img_per_class,
                          thresold,
                          modelfn,
                          batch_size=10):
    while(True):
        anchorlist=[]
        positivelist=[]
        negativelist=[]
        for i in range(batch_size):
            pair=np.random.randint(0,len(kindpathlist),n_class_per_batch)
            imgpathlist=[]
            for index1 in pair:
                labelpathlist=kindpathlist[index1]
                if(len(labelpathlist)>n_img_per_class):
                    labelpair=np.random.randint(0,len(labelpathlist),n_img_per_class)
                    for index1 in labelpair:
                        imgpathlist.append(labelpathlist[index1])
                else:
                    num1=n_class_per_batch
                    while(True):
                        for imgpath in labelpathlist:
                            if(num1>0):
                                imgpathlist.append(imgpath)
                                num1-=1
                            else:
                                break

            imglist=fetch_path_imglist(BASE_DIR,IMG_ROW,IMG_COL,imgpathlist)
            featurelist=modelfn.predict(imglist)
            distances=cosine_distance(featurelist)
            anchor_id,pos_id,neg_id=semi_hard_mining(distances,
                                                     n_class_per_batch,
                                                     n_img_per_class,
                                                     thresold)
            