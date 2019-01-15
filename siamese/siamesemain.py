import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from imggen.siamese.siameseimggen import siamese_train_gen,siamese_valid_gen
from model.siamese.siamesenet import siamsenet

BASE_DIR='../train/'

IMG_ROW=IMG_COL=256
num_thresold=30
labelpath='train.csv'
traindata=pd.read_csv(labelpath)
modelfn=DenseNet121(include_top=False,
                    weights=None)
pathdata=traindata['Image']
labeldata=traindata['Id']

#pretrained_model=pretrainedmodel(IMG_ROW,IMG_COL,modelfn,labelnum)
#pretrained_model.load_weights('model/models/pretrainmodel.h5')
'''pretrained_model1=Sequential()
for layer in pretrained_model.layers[:-3]:
    pretrained_model1.add(layer)'''
model=siamsenet(IMG_ROW,IMG_COL,modelfn)
model.summary()
model.compile(optimizer=Adam(0.001),metrics=['accuracy'],
              loss=['binary_crossentropy'])
callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('model/models/siamese.h5',monitor='val_loss',save_best_only=True,verbose=1)
]

train_pathdata,valid_pathdata,train_labeldata,valid_labeldata=train_test_split(pathdata,labeldata,
                                                                               test_size=0.1)
validdata=siamese_valid_gen(BASE_DIR,IMG_ROW,IMG_COL,valid_pathdata,valid_labeldata)
history=model.fit_generator(siamese_train_gen(BASE_DIR,IMG_ROW,IMG_COL,train_pathdata,train_labeldata,batch_size=50),
                            steps_per_epoch=100,
                            epochs=100,
                            validation_data=validdata,
                            callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
plt.show()
