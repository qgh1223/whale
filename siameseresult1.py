from result.siameseresultgen1 import similiarityarr,fetch_test_imglist
from preprocess.kindlist import fetch_all_kind_list,fetch_kind_list_split,fetch_all_kind_img_list
import pandas as pd
import os
from keras.models import load_model

IMG_ROW=IMG_COL=64
train_dir='train/'
test_dir='test/'
labelpath='train.csv'
traindata=pd.read_csv(labelpath)

_,kindlist,kindimglist=fetch_all_kind_img_list(train_dir,IMG_ROW,IMG_COL,traindata)
test_imglist=fetch_test_imglist(test_dir,IMG_ROW,IMG_COL)
predictlabellist=[]
print(len(test_imglist))
model=load_model('siamese.h5')

for i,test_img in enumerate(test_imglist):
    print(i)
    similiaritystr=similiarityarr(train_dir,IMG_ROW,IMG_COL,test_img,kindimglist,kindlist,model)
    predictlabellist.append(similiaritystr)

resultcsv=pd.DataFrame({
    'Image':os.listdir(test_dir),
    'Id':predictlabellist
})
resultcsv.to_csv('result.csv',index=False)