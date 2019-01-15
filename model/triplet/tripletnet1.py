from keras.layers import *
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import keras.backend as K

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a-p), axis=-1)
        n_dist = K.sum(K.square(a-n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(TripletLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def tripletnet(IMG_ROW,IMG_COL,modelfn):
    inputlayerlist=[]
    featurelist=[]
    for i in range(3):
        inputlayer=Input((IMG_ROW,IMG_COL,3))
        x=modelfn(inputlayer)
        inputlayerlist.append(inputlayer)
        x=GlobalAveragePooling2D()(x)
        featurelist.append(x)
    triplet_loss_layer=TripletLossLayer(alpha=0.05)(featurelist)
    model=Model(inputlayerlist,triplet_loss_layer)
    return model
