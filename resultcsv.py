import pandas as pd
import os
import numpy as np
resultcsv=pd.read_csv('result (2).csv')
labelstrarr=[]
for labelstr in resultcsv['Id']:
    labelarr=labelstr.split(' ')
    uniquelabel=np.unique(labelarr)
    if(len(uniquelabel)<5):
        labelarr1=[]
        for i in range(6-len(uniquelabel)):
            labelarr1.append(uniquelabel[0])
        for j in range(1,len(uniquelabel)):
            labelarr1.append(uniquelabel[j])
        print(labelarr1)
        labelstrarr.append(' '.join(labelarr1))
    else:
        labelstrarr.append(' '.join(uniquelabel[:5]))
resultcsv=pd.DataFrame({
    'Image':resultcsv['Image'],
    'Id':labelstrarr
})
print(resultcsv['Id'][0])
resultcsv.to_csv('resultcsv1.csv',index=False)