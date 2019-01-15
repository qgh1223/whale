from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from model.pretrainedmodel import pretrainedmodel
from imggen.pretrainimggen import pretrain_train_gen,pretrain_valid_gen
from preprocess.pretrainpreprocess import preprocess,img_all_list,imggen
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
IMG_ROW=IMG_COL=256
num_thresold=30
BASE_DIR='train/'
labelpath='train.csv'
traindata=pd.read_csv(labelpath)

pathdata,labeldata,labelnum=preprocess(traindata,
                                       num_thresold)
train_pathdata,valid_pathdata,train_labeldata,valid_labeldata=train_test_split(traindata['Image'],traindata['Id'],
                                                                               test_size=0.1)
modelfn=DenseNet121(include_top=False,input_shape=(IMG_ROW,IMG_COL,3),
                    weights=None)

model=pretrainedmodel(IMG_ROW,IMG_COL,modelfn,1)
model.summary()
model.compile(optimizer=Adam(0.001),metrics=['accuracy'],
              loss=['binary_crossentropy'])
callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('model/models/pretrainmodel.h5',monitor='val_loss',save_best_only=True,verbose=1)
]
validdata=pretrain_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,valid_pathdata,valid_labeldata)
history=model.fit_generator(pretrain_train_gen(BASE_DIR,IMG_ROW,IMG_COL,train_pathdata,
                                               train_labeldata,batch_size=50),
                            steps_per_epoch=100,
                            epochs=100,
                            validation_data=validdata,
                            callbacks=callbacks)
