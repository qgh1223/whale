import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
traindata=pd.read_csv('train.csv')
kindlist=traindata.groupby('Id').size()[1:]
plt.hist(list(kindlist),100,normed=1,histtype='bar',facecolor='red',alpha=0.75)
plt.show()