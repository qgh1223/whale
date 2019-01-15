from result.siameseresultgen import similiarityarr,fetch_test_imglist,fetch_train_imglist
import pandas as pd
import os
IMG_ROW=IMG_COL=64
train_dir='train/'
test_dir='test/'
csvpath='train.csv'
train_imglist,labellist=fetch_train_imglist(train_dir,csvpath,IMG_ROW,IMG_COL)
test_imglist=fetch_test_imglist(test_dir,IMG_ROW,IMG_COL)
predictlabellist=[]
print(len(test_imglist))
for i,test_img in enumerate(test_imglist):
    print(i)
    similiaritystr=similiarityarr(test_img,train_imglist,labellist,'siamese.h5')
    predictlabellist.append(similiaritystr)
resultcsv=pd.DataFrame({
    'Image':os.listdir(train_dir),
    'Id':predictlabellist
})
resultcsv.to_csv('result.csv',index=False)