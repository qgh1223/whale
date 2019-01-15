import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam

from imggen.triplet.tripletimggen2 import triplet_img_gen
from model.triplet import tripletnet
from preprocess.kindlist import fetch_all_kind_list1,fetch_kind_list_split

BASE_DIR='../train/'

IMG_ROW=IMG_COL=64
labelpath='../train.csv'
traindata=pd.read_csv(labelpath)
modelfn=MobileNetV2(include_top=False,weights=None)
modelfn.load_weights('../mobile_encoder.h5')
model=tripletnet.triplet_net(IMG_ROW,IMG_COL,modelfn)
model.summary()
model.compile(loss=None, optimizer=Adam(0.0002))
callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('model/models/triplet.h5',monitor='val_loss',save_best_only=True,verbose=1)
]
kindimgpathlist,kindlist=fetch_all_kind_list1(traindata)
trainkindimgpathlist,validkindimgpathlist=fetch_kind_list_split(kindimgpathlist)
history=model.fit_generator(triplet_img_gen(BASE_DIR,IMG_ROW,IMG_COL,trainkindimgpathlist),
                  epochs=100,
                  validation_data=triplet_img_gen(BASE_DIR,IMG_ROW,IMG_COL,validkindimgpathlist,batch_size=5),
                  validation_steps=10,
                  steps_per_epoch=30,
                  callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
plt.show()
