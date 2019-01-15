import numpy as np
import pandas as pd
import cv2
import random
import matplotlib.pyplot as plt
def imgarr(imgpath):
    img=cv2.imread(imgpath)
    return img

def siamese_fetch_img(BASE_DIR,label,labeldata,pathdata,kind):
    if(kind==0):
        pathdata1=pathdata[labeldata!=label]

    else:
        pathdata1=pathdata[labeldata==label]
    rndid=random.randint(0,len(pathdata1)-1)
    path=pathdata1[pathdata1.index[rndid]]
    img=imgarr(BASE_DIR+path)
    return img

