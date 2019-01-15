import numpy as np
import random
import cv2
from imggen.fetchimg import imgarr,siamese_fetch_img

def siamese_img_gen(BASE_DIR,IMG_ROW,IMG_COL,kindimgpathlist,
                    contrast_times=10,batch_size=50):
    while(True):
        imglist1=[]
        imglist2=[]
        labellist=[]
        for i in range(batch_size):
            for j in range(contrast_times):
                rndid=random.randint(0,len(kindimgpathlist)-1)
                if(i%2==0):
                    pair=np.random.randint(0,len(kindimgpathlist[rndid]),2)
                    imgpath1=kindimgpathlist[rndid][pair[0]]
                    imgpath2=kindimgpathlist[rndid][pair[1]]
                    labellist.append(0)
                else:
                    rndid1=random.randint(0,len(kindimgpathlist[rndid])-1)
                    imgpath1=kindimgpathlist[rndid][rndid1]
                    index1=random.choice([num for num in range(len(kindimgpathlist)) if num not in [rndid]])
                    rndid2=random.randint(0,len(kindimgpathlist[index1])-1)
                    imgpath2=kindimgpathlist[index1][rndid2]
                    labellist.append(1)
                img1=imgarr(BASE_DIR+imgpath1)
                img2=imgarr(BASE_DIR+imgpath2)
                img1=cv2.resize(img1,(IMG_ROW,IMG_COL))
                img2=cv2.resize(img2,(IMG_ROW,IMG_COL))
                imglist1.append(img1)
                imglist2.append(img2)
        yield ([np.asarray(imglist1),np.asarray(imglist2)],np.asarray(labellist))
