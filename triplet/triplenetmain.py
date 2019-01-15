import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from imggen.triplet.tripletimggen import triplet_train_gen,triplet_valid_gen
from model.triplet import tripletnet

BASE_DIR='../train/'

IMG_ROW=IMG_COL=256
labelpath='../train.csv'
traindata=pd.read_csv(labelpath)
modelfn=InceptionV3(include_top=False,
                    weights=None)
pathdata=traindata['Image']
labeldata=traindata['Id']

model=tripletnet.tripletnet(IMG_ROW,IMG_COL,modelfn)
model.summary()
model.compile(loss=None, optimizer=Adam(0.0002))

callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('model/models/triplet.h5',monitor='val_loss',save_best_only=True,verbose=1)
]

train_pathdata,valid_pathdata,train_labeldata,valid_labeldata=train_test_split(pathdata,labeldata,
                                                                               test_size=0.1)
print(valid_labeldata)
validdata=triplet_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,valid_pathdata,valid_labeldata)
history=model.fit_generator(triplet_train_gen(BASE_DIR,IMG_ROW,IMG_COL,train_pathdata,train_labeldata,batch_size=30),
                            steps_per_epoch=100,
                            epochs=100,
                            validation_data=validdata,
                            callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
plt.show()

