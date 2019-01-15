import numpy as np
import random
import cv2
from imggen.fetchimg import imgarr,siamese_fetch_img

def siamese_img_gen(BASE_DIR,IMG_ROW,IMG_COL,kindimglist,
                    contrast_times=10,batch_size=50):
    while(True):
        imglist1=[]
        imglist2=[]
        labellist=[]
        for i in range(batch_size):
            for j in range(contrast_times):
                rndid=random.randint(0,len(kindimglist)-1)
                if(i%2==0):
                    pair=np.random.randint(0,len(kindimglist[rndid]),2)
                    img1=kindimglist[rndid][pair[0]]
                    img2=kindimglist[rndid][pair[1]]
                    labellist.append(0)
                else:
                    rndid1=random.randint(0,len(kindimglist[rndid])-1)
                    img1=kindimglist[rndid][rndid1]
                    index1=random.choice([num for num in range(len(kindimglist)) if num not in [rndid]])
                    rndid2=random.randint(0,len(kindimglist[index1])-1)
                    img2=kindimglist[index1][rndid2]
                    labellist.append(1)
                img1=cv2.resize(img1,(IMG_ROW,IMG_COL))
                img2=cv2.resize(img2,(IMG_ROW,IMG_COL))
                imglist1.append(img1)
                imglist2.append(img2)
        yield ([np.asarray(imglist1),np.asarray(imglist2)],np.asarray(labellist))
