import pandas as pd
import numpy as np

def similiarity(img1,img2,model):
    result=model.predict([[img1],[img2]])
    return result[0][0]

def similiarity_index(img,imglist,model,labeldata):
    similiaritylist=[]
    for img1 in imglist:
        similiarityvalue=similiarity(img,img1,model)
        similiaritylist.append(similiarityvalue)
    maxvalue=np.max(similiaritylist)
    index1=similiaritylist.index(maxvalue)
    return labeldata[index1]

def labelpredict(imglist1,imglist2,labellist,model):
    labelpredictlist=[]
    for img in imglist1:
        label=similiarity_index(img,imglist2,model,labellist)
        labelpredictlist.append(label)
    