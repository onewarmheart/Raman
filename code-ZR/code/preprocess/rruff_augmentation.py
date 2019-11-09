# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:11:37 2018

@author: ZR
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


NUM_OF_SHIFT_PER_IMAGE=4
NUM_OF_NOISE_PER_IMAGE=1
SHIFT=2
SIGMA=1.2


#return: 不包含原有的X，只有新生成的
def combine(Y,C):
    
    Y_pre=[]
    C_pre=[]
    for i in range(len(C)-1):
        y_aug = []        
        if C[i]==C[i+1]:
            for j in range(len(Y[i])):
                a = random.random()
                y = Y[i][j]*a + Y[i+1][j]*(1-a)
                y_aug.append(y)
            Y_pre.append(y_aug)
            C_pre.append(C[i])
    return np.array(Y_pre), np.array(C_pre)
    

 
# return: X, X[i]_1...X[i]_n(i:0~len(X))
def shift(X,Y,C,N):
    
    X_pre=[]
    Y_pre=[]
    C_pre=[]
    
    for m in range(len(X)):
        X_pre.append(X[m])
        Y_pre.append(Y[m])
        C_pre.append(C[m])
    #print(X_pre)
    for i in range(len(X)):
        if N[int(C[i])] <= 10:
            num = int(round((10-N[int(C[i])]) /  N[int(C[i])]))
            for n in range(num):    
                X_pre.append(X[i])
                Y_pre.append(Y[i])
                C_pre.append(C[i])
    return np.array(X_pre), np.array(Y_pre), np.array(C_pre)


def noise(X,Y,C,N):
    
    X_pre=[]
    Y_pre=[]
    C_pre=[]
    for m in range(len(X)):
        X_pre.append(X[m])
        Y_pre.append(Y[m])
        C_pre.append(C[m])
        
    for i in range(len(X)):
        num = int(round((20 - N[int(C[i])]) / N[int(C[i])]))
        for n in range(num):
            X_pre.append(X[i])
            Y_pre.append(Y[i])
            C_pre.append(C[i])
            
    return np.array(X_pre), np.array(Y_pre),  np.array(C_pre)


if __name__ == '__main__':
    
    path='/home/zr/AI/rruff'
    dire='ex_un'
    
# =============================================================================
#     np.set_printoptions(threshold=np.inf)
#     data_x = np.load('%s/data/%s/remove_1/uninter_xx_train.npy'%(path,dire))
#     data_y = np.load('%s/data/%s/remove_1/uninter_xy_train.npy'%(path,dire))
#     data_c = np.load('%s/data/%s/remove_1/yclass_train.npy'%(path,dire))
#     
#     X_,Y_,C_ = shift(data_x,data_y,data_c)
#     print(X_.shape)
#     X,Y,C= noise(X_,Y_,C_)
#     print(X.shape)
#     np.save('%s/data/%s/copy/uninter_xx_train.npy'%(path,dire), X)
#     np.save('%s/data/%s/copy/uninter_xy_train.npy'%(path,dire), Y)
#     np.save('%s/data/%s/copy/yclass_train.npy'%(path,dire), C)
# =============================================================================
    classnum = np.load('%s/data/%s/remove_1/classnum_train.npy'%(path,dire))
    data_x = np.load('%s/data/%s/remove_1/uninter_xx_train.npy'%(path,dire))
    data_y = np.load('%s/data/%s/remove_1/uninter_xy_train.npy'%(path,dire))
    data_c = np.load('%s/data/%s/remove_1/yclass_train.npy'%(path,dire))
    print(data_x.shape,len(classnum))
    X_,Y_,C_= shift(data_x,data_y,data_c,classnum)
    
# =============================================================================
#     C_ = list(C_)
#     C_.sort()
#     C_ = np.array(C_)
#     c=0
#     num=0
#     Num=[]
#     for i in range(len(C_)):
#         if C_[i]==c:
#             num+=1
#         else:
#             c+=1
#             Num.append(num)
#             num=1
#     Num.append(num)
#     Num=np.array(Num)
#     print(Num,Num.shape)
#     np.save('%s/data/%s/copy/classnum_shift.npy'%(path,dire), Num)
# =============================================================================
    
    
    num = np.load('/home/zr/AI/rruff/data/ex_un/copy/classnum_shift.npy')
    print(X_.shape,Y_.shape,C_.shape)
    X,Y,C= noise(X_,Y_,C_,num)
    print(X.shape,Y.shape,C.shape)
    np.save('%s/data/%s/copy/uninter_xx_train.npy'%(path,dire), X)
    np.save('%s/data/%s/copy/uninter_xy_train.npy'%(path,dire), Y)
    np.save('%s/data/%s/copy/yclass_train.npy'%(path,dire), C)

    C = list(C)
    C.sort()
    C = np.array(C)
    c=0
    num=0
    Num=[]
    for i in range(len(C)):
        if C[i]==c:
            num+=1
        else:
            c+=1
            Num.append(num)
            num=1
    Num.append(num)
    Num=np.array(Num)
    print(Num,Num.shape)
    np.save('%s/data/%s/copy/classnum_aug.npy'%(path,dire), Num)
    

    