import cv2
import numpy as np
from keras.preprocessing.image import img_to_array,load_img
img=img_to_array(load_img('1546172519(1).png',grayscale=True,target_size=(256,256)))
print(img.shape)
labels,index1=np.unique(img,return_index=True)
print(labels)
print(index1)