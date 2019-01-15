import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
train_dir='train/'
test_dir='test/'

def fileexists(path):
    if(os.path.exists(train_dir+path)):
        return train_dir+path
    elif(os.path.exists(test_dir+path)):
        return test_dir+path

boundboxdata=pd.read_csv('bounding_boxes.csv')
for i in range(9):
    plt.subplot(str(331+i))
    rnd1=random.randint(0,len(boundboxdata)-1)
    path=boundboxdata['Image'][rnd1]
    imgpath=fileexists(path)
    img=cv2.imread(imgpath)
    x1=boundboxdata['x0'][rnd1]
    y1=boundboxdata['y0'][rnd1]
    x2=boundboxdata['x1'][rnd1]
    y2=boundboxdata['y1'][rnd1]
    img=cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))
    plt.imshow(img)
