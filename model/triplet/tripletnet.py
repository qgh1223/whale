from keras.layers import *
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import keras.backend as K
def triplet_loss(inputs,margin=0.05):
    anchor,positive,negative=inputs
    positive_distance=K.square(anchor-positive)
    negative_distance=K.square(anchor-negative)
    positive_distance=K.sum(positive_distance,axis=-1,keepdims=True)
    negative_distance=K.sum(negative_distance,axis=-1,keepdims=True)
    loss=positive_distance-negative_distance
    loss = K.log(1 + K.exp(loss))
    return K.mean(loss)

def triplet_net(IMG_ROW,IMG_COL,modelfn):
    inputlayerlist=[]
    featurelist=[]
    for i in range(3):
        inputlayer=Input((IMG_ROW,IMG_COL,3))
        x=modelfn(inputlayer)
        inputlayerlist.append(inputlayer)
        x=GlobalAveragePooling2D()(x)
        x=Dense(50,activation='relu')(x)
        featurelist.append(x)
    model=Model(inputlayerlist,featurelist)
    model.add_loss(K.mean(triplet_loss(featurelist)))
    return model
