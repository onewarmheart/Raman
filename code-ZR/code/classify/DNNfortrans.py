# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 21:02:19 2018

@author: ZR
"""

import numpy as np
import os
import json
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

BATCH_SIZE=32
EPOCHS=50
LAYERS=7
VOCAB_SIZE=72 #72
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
    
    model.add(Dense(3072, input_shape=(1024,),name="layer_1"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    
    model.add(Dense(2048, input_shape=(1024,),name="layer_2"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    
    model.add(Dense(2048, input_shape=(1024,),name="layer_3"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    
    for i in range(LAYERS-5):
        a=i+4
        model.add(Dense(2048,name="layer_%d"%a))
        model.add(Activation('tanh'))
        model.add(Dropout(0.4))  
    
    model.add(Dense(1024,name="layer1_6"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    
    model.add(Dense(VOCAB_SIZE,name="layer1_7"))
    model.add(Activation('softmax'))
    
    model.compile(optimizer=Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=5),
                  loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


def Train_pre():
    
    
    for i in range(EPOCHS):
        print('epochs:',i)
        #hist=model.fit_generator(generator=generater('train'), steps_per_epoch=STEPS_PER_EPOCH, 
                                 #epochs=1,validation_data=generater('val'),validation_steps=1)
        hist=model.fit_generator(generator=generater('train'), steps_per_epoch=STEPS_PER_EPOCH,epochs=1)
        model.save_weights('%s/modelorganic/DNN.%d.h5'%(path,i), overwrite=True)
        model.save_weights('%s/modelorganic/DNN.h5'%path, overwrite=True) 
       
def Transfer(transfer=True,level = 0):
    
    if transfer == True:       
        model.load_weights('/home/zr/AI/pretrain/modelrruff/DNN.h5',by_name=True)
        
# =============================================================================
#     if level == 0: 
#         for i,layer in enumerate(model.layers):
#             if i==0:
#                 layer.trainable = False
# =============================================================================

                
    model.compile(optimizer=Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=5),
                  loss='categorical_crossentropy',metrics=['accuracy']) 
    
    
         
    for i in range(EPOCHS):
        print('epochs:',i)
        hist=model.fit(X[train],Y[train],validation_data=(X[val],Y[val]),epochs=1)
        with open('%s/res/rruff/DNN/10DNN%dlayers%depochs0.00005lr.npy'
                  %(path, LAYERS,EPOCHS), 'a') as f:
                    json.dump(hist.history, f)
                    f.write('\n')
    cvscores.append(hist.history['val_acc'])
        
       
if __name__ == '__main__':

    path='/home/zr/AI/trans'
    
    X_train = np.load('%s/data/rruff/xy_train.npy'%path)
    Y_train = np.load('%s/data/rruff/yclass_train.npy'%path)
    X_val = np.load('%s/data/rruff/xy_val.npy'%path)
    Y_val_ = np.load('%s/data/rruff/yclass_val.npy'%path)
    #X_train = np.load('%s/data/xy.npy'%path)
    #Y_train = np.load('%s/data/yclass.npy'%path)
    STEPS_PER_EPOCH=int(len(X_train)/BATCH_SIZE)
    #Y_train = vocab(Y_train)
    #Y_val = vocab(Y_val_)
    Y_train = vocab(Y_train)
    print(Y_train.shape)
    model = build_model()
    Train_pre()
    

    X = np.load('%s/data/scale_xy.npy'%path) 
    Y_ = np.load('%s/data/yclass.npy'%path) 
    
    skf = StratifiedKFold(n_splits=3)    
    cvscores = []
    for train, val in skf.split(X, Y_):
        model=build_model()
        Y = vocab(Y_)
        Transfer(transfer=True)                
    print("%.5f%% (+/- %.5f%%)" % (np.mean(cvscores), np.std(cvscores)))
    #X = np.concatenate((X_train,X_val),axis=0)
    #Y = np.concatenate((Y_train,Y_val),axis=0)