from keras.models import load_model
import numpy as np
import cv2
class NewImgGenerate:
    def __init__(self,imgpath,modelpath,IMG_ROW,IMG_COL):
        self.imgpath=imgpath
        self.modelpath=modelpath
        self.IMG_ROW=IMG_ROW
        self.IMG_COL=IMG_COL

    def position_arr(self):
        model=load_model(self.modelpath)
        img=cv2.imread(self.imgpath)
        img=cv2.resize(img,(self.IMG_ROW,self.IMG_COL))
        self.img=img
        imglist=[img]
        positionpredictarr=model.predict(imglist)
        return positionpredictarr

    def resize_new_img(self):
        positionarr=self.position_arr()
        xmin=int(positionarr[0][0])
        ymin=int(positionarr[0][1])
        xmax=int(positionarr[0][2])
        ymax=int(positionarr[0][3])
        newimgarr=np.zeros((xmax-xmin,ymax-ymin,3))
        for i in range(xmin,xmax):
            for j in range(ymin,ymax):
                for k in range(3):
                    newimgarr[i-xmin][j-ymin][k]=self.img[i][j][k]
        newimgarr=cv2.resize(newimgarr,(self.IMG_ROW,self.IMG_COL,3))
        return newimgarr