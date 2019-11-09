# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:00:37 2018

@author: ZR
"""

import numpy as np
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import random
import json
import matplotlib.pyplot as plt
import os
from keras import backend as K
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

BATCH_SIZE=32
EPOCHS=100
LAYERS=5
VOCAB_SIZE=72
LR=0.00005



def vocab(y):
    data_y=np.zeros((len(y),VOCAB_SIZE))
    for i,j in enumerate(y):
        data_y[i][int(j)]=1
    return data_y
            
def generater(mode,batch_size=BATCH_SIZE):
    
    cnt = 0
    X = []
    Y = []
    while 1:
        if mode == 'train':
            idex = random.randint(0,len(X_train)-1)
            X.append(X_train[idex])
            Y.append(Y_train[idex])
        if mode == 'val':
            X = X_val
            Y = Y_val
        cnt += 1
        if cnt == batch_size:
            yield np.array(X), np.array(Y)
            cnt = 0
            X = []
            Y = []          
    
def build_model(batch_size=BATCH_SIZE):
    
    model=Sequential()
    
    model.add(Convolution1D(16,21, border_mode='same', batch_input_shape=(None,1024,1),name="layer_1"))
    model.add(LeakyReLU(alpha=.3))
    model.add(MaxPooling1D(pool_size=2,strides=2,padding='same'))
    model.add(Dropout(0.4))
    
    model.add(Convolution1D(32,11, border_mode='same',name="layer_2"))
    model.add(LeakyReLU(alpha=.3))
    model.add(MaxPooling1D(pool_size=2,strides=2,padding='same'))
    model.add(Dropout(0.4))
    
    model.add(Convolution1D(64,5, border_mode='same',name="layer1_3"))
    model.add(LeakyReLU(alpha=.3))
    model.add(MaxPooling1D(pool_size=2,strides=2,padding='same'))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(2048,name="layer1_4"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    
    model.add(Dense(VOCAB_SIZE,name="layer1_5"))
    model.add(Activation('softmax'))
    
    model.compile(optimizer=Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=5),
                  loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def Train_pre():

    for i in range(EPOCHS):
        print('epochs:',i)
        hist=model.fit_generator(generator=generater('train'), steps_per_epoch=STEPS_PER_EPOCH,epochs=1)
        model.save_weights('%s/CNN/model2/CNN.%d.h5'%(path,i), overwrite=True)
        model.save_weights('%s/CNN/model2/CNN.h5'%path, overwrite=True) 

def Transfer(transfer=True,level = 0):
    
    if transfer == True:       
        model.load_weights('/home/zr/AI/pretrain/CNN/model/CNN.h5',by_name=True)
                        
    model.compile(optimizer=Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=5),
                  loss='categorical_crossentropy',metrics=['accuracy']) 
            
    for i in range(EPOCHS):
        print('epochs:',i)
        hist=model.fit(X[train],Y[train],validation_data=(X[val],Y[val]),epochs=1)
        with open('%s/res/organic/CNN/11CNN%dlayers%depochs0.00005lr.npy'
                  %(path, LAYERS,EPOCHS), 'a') as f:
                    json.dump(hist.history, f)
                    f.write('\n')
    cvscores.append(hist.history['val_acc'])
        
if __name__ == '__main__':

    path='/home/zr/AI/trans'

# =============================================================================
#     #X_train = np.load('%s/data/rruff/xy_train.npy'%path)
#     #Y_train = np.load('%s/data/rruff/yclass_train.npy'%path)
#     #X_val = np.load('%s/data/rruff/xy_val.npy'%path)
#     #Y_val_ = np.load('%s/data/rruff/yclass_val.npy'%path)
#     X_train = np.load('%s/data/xy.npy'%path)
#     Y_train = np.load('%s/data/yclass.npy'%path)
#     X_train = np.reshape(X_train,(-1,1024,1))
#     #X_val = np.reshape(X_val,(-1,1024,1))
#     Y_train = vocab(Y_train)
#     #Y_val = vocab(Y_val_)
#     STEPS_PER_EPOCH=int(len(X_train)/BATCH_SIZE)
#     
#     print(Y_train.shape)
#     model = build_model()
#     Train_pre()
# =============================================================================

    X_ = np.load('%s/data/scale_xy.npy'%path) 
    Y_ = np.load('%s/data/yclass.npy'%path) 
    
    skf = StratifiedKFold(n_splits=3)    
    cvscores = []
    for train, val in skf.split(X_, Y_):
        
        X = np.reshape(X_,(-1,1024,1))
        Y = vocab(Y_)
        model=build_model()
        Transfer(transfer=True)                
    print("%.5f%% (+/- %.5f%%)" % (np.mean(cvscores), np.std(cvscores)))
    

