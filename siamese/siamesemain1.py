import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam

from imggen.siamese.siameseimggen1 import siamese_img_gen
from model.siamese.siamesenet import siamsenet
from preprocess.kindlist import fetch_all_kind_list,fetch_kind_list_split

BASE_DIR='train/'

IMG_ROW=IMG_COL=256
labelpath='train.csv'
traindata=pd.read_csv(labelpath)
modelfn=MobileNetV2(include_top=False,
                    weights=None)
pathdata=traindata['Image']
labeldata=traindata['Id']
model=siamsenet(IMG_ROW,IMG_COL,modelfn)
model.summary()
model.compile(optimizer=Adam(0.001),metrics=['accuracy'],
              loss=['binary_crossentropy'])
callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('../model/models/siamese.h5',monitor='val_loss',save_best_only=True,verbose=1)
]

kindimgpathlist,kindlist=fetch_all_kind_list(traindata)
trainkindimgpathlist,validkindimgpathlist=fetch_kind_list_split(kindimgpathlist)
history=model.fit_generator(siamese_img_gen(BASE_DIR,IMG_ROW,IMG_COL,
                                            trainkindimgpathlist,batch_size=50),
                            steps_per_epoch=50,
                            epochs=100,
                            validation_data=siamese_img_gen(BASE_DIR,IMG_ROW,IMG_COL,
                                                            validkindimgpathlist,contrast_times=10,batch_size=5),
                            validation_steps=20,
                            callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
plt.show()
modelfn.save('res_encoder.h5')