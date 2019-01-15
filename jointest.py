import pandas as pd
traindata=pd.read_csv('train.csv')
bounddata=pd.read_csv('bounding_boxes.csv')
print(pd.merge(traindata,bounddata))