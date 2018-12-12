from keras.layers import *
from keras.models import Model

def pretrainedmodel(IMG_ROW,IMG_COL,modelfn,LABELNUM):
    inputlayer=Model((IMG_ROW,IMG_COL,3))
    feature=modelfn(inputlayer)
    feature=Dense(256,activation='relu')(feature)
    feature=BatchNormalization()(feature)
    feature=Dense(LABELNUM,activation='softmax')(feature)
    return Model(inputlayer,feature)

