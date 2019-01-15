import pandas as pd
import numpy as np
import random
import cv2
from imggen.fetchimg import imgarr
def triplet_img_gen(BASE_DIR,IMG_ROW,IMG_COL,kindlist,kindimgpathlist,triplet_num=40):
    tripletlist=[]

    for i in range(len(kindimgpathlist)):
        print(i)
        for j in range(triplet_num):
            negativeindex=random.choice([num for num in range(len(kindimgpathlist)) if num not in [i]])
            pair=np.random.randint(0,len(kindimgpathlist[i]),2)
            anchorpath=kindimgpathlist[i][pair[0]]
            anchor=imgarr(BASE_DIR+anchorpath)
            positivepath=kindimgpathlist[i][pair[1]]
            positive=imgarr(BASE_DIR+positivepath)
            rnd_id=random.randint(0,len(kindimgpathlist[negativeindex])-1)
            negativepath=kindimgpathlist[negativeindex][rnd_id]
            negative=imgarr(BASE_DIR+negativepath)
            anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
            positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
            negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
            triplet=[anchor,positive,negative]
            tripletlist.append(triplet)
    return np.array(tripletlist)

def triplet_img_gen1(BASE_DIR,IMG_ROW,IMG_COL,kindlist,kindimgpathlist,times_kind=40,
                     triplet_num=10,batch_size=10):
    while(True):
        anchorlist=[]
        positivelist=[]
        negativelist=[]
        for i in range(batch_size):
            print('batch_size:'+str(i))
            for j in range(times_kind):
                print('kind_num:'+str(j))
                for k in range(triplet_num):
                    rndid=random.randint(0,len(kindimgpathlist)-1)
                    negativeindex=random.choice([num for num in range(len(kindimgpathlist)) if num not in [rndid]])
                    pair=np.random.randint(0,len(kindimgpathlist[rndid]),2)
                    anchorpath=kindimgpathlist[rndid][pair[0]]
                    positivepath=kindimgpathlist[rndid][pair[1]]
                    rndid1=random.randint(0,len(kindimgpathlist[negativeindex])-1)
                    negativepath=kindimgpathlist[negativeindex][rndid1]
                    anchor=imgarr(BASE_DIR+anchorpath)
                    positive=imgarr(BASE_DIR+positivepath)
                    negative=imgarr(BASE_DIR+negativepath)
                    anchor=cv2.resize(anchor,(IMG_ROW,IMG_COL))
                    positive=cv2.resize(positive,(IMG_ROW,IMG_COL))
                    negative=cv2.resize(negative,(IMG_ROW,IMG_COL))
                    anchorlist.append(anchor)
                    positivelist.append(positive)
                    negativelist.append(negative)
        yield ([np.asarray(anchorlist),np.asarray(positivelist),
                np.asarray(negativelist)],
               None)