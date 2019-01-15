from keras.layers import *
from keras.models import Model

def siamsenet(IMG_ROW,IMG_COL,modelfn):

    input_a=Input((IMG_ROW,IMG_COL,3))

    input_b=Input((IMG_ROW,IMG_COL,3))

    feature_a=modelfn(input_a)

    feature_b=modelfn(input_b)
    feature_a=GlobalAveragePooling2D()(feature_a)
    feature_b=GlobalAveragePooling2D()(feature_b)

    combined_feature=concatenate([feature_a,feature_b])

    combined_feature=Dense(16,activation='relu')(combined_feature)
    combined_feature=BatchNormalization()(combined_feature)
    combined_feature=Activation('relu')(combined_feature)
    combined_feature=Dense(1,activation='sigmoid')(combined_feature)
    model=Model([input_a,input_b],combined_feature)
    return model

