from model.siamese.lambdasiamese import build_siamese_model
from imggen.siamese.siameseimggen1 import siamese_img_gen
from preprocess.kindlist import fetch_all_kind_list,fetch_kind_list_split
from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
BASE_DIR='train/'
IMG_ROW=IMG_COL=64
img_shape=(IMG_ROW,IMG_COL,3)
branch_model=MobileNetV2(weights=None,
                         input_shape=(IMG_ROW,IMG_COL,3),
                         classes=200)

model,head_model = build_siamese_model(img_shape,64e-5,branch_model)

labelpath='train.csv'
traindata=pd.read_csv(labelpath)
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
