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


NUM_OF_SHIFT_PER_IMAGE=6
NUM_OF_NOISE_PER_IMAGE=1
SHIFT=0.2
SIGMA=0.00001

path='/home/zr/AI/pretrain'
 
# return: X, X[i]_1...X[i]_n(i:0~len(X))
def shift(X,Y,C):
    
    X_pre=[]
    Y_pre=[]
    C_pre=[]
    length_x=SHIFT
    for m in range(len(X)):
        X_pre.append(X[m])
        Y_pre.append(Y[m])
        C_pre.append(C[m])
    #print(X_pre)
    for i in range(len(X)):
        x=0
        for n in range(NUM_OF_SHIFT_PER_IMAGE):
            x_aug=[]
            for j in range(len(X[i])): 
                if n%2==0:
                    x = X[i][j] + length_x * (n/2+1)
                    x_aug.append(x)
                else:
                    x = X[i][j] - length_x * ((n-1)/2+1)
                    x_aug.append(x)
            X_pre.append(x_aug)
            Y_pre.append(Y[i])
            C_pre.append(C[i])
    return np.array(X_pre), np.array(Y_pre), np.array(C_pre)
    

def noise(X,Y,C):
    
    X_pre=[]
    Y_pre=[]
    C_pre=[]
    mu = 0
    sigma = SIGMA
    for m in range(len(X)):
        X_pre.append(X[m])
        Y_pre.append(Y[m])
        C_pre.append(C[m])
        
    for i in range(len(X)):
        for n in range(NUM_OF_NOISE_PER_IMAGE):
            x_aug=[]
            y_aug=[]
            for j in range(len(X[i])):           
                x = X[i][j] + random.gauss(mu,sigma)
                y = Y[i][j] + random.gauss(mu,sigma)
                x_aug.append(x)
                y_aug.append(y)
            X_pre.append(x_aug)
            Y_pre.append(y_aug)
            C_pre.append(C[i])
            
    return np.array(X_pre), np.array(Y_pre),  np.array(C_pre)

if __name__ == '__main__':
    
    np.set_printoptions(threshold=np.inf)
    data_x = np.load('%s/data/xx.npy'%path)
    data_y = np.load('%s/data/xy.npy'%path)
    data_c = np.load('%s/data/yclass.npy'%path)
    X_,Y_,C_= shift(data_x,data_y,data_c)
    print(C_,X_.shape,C_.shape)
    X,Y,C= noise(X_,Y_,C_)
    np.save('%s/data/shift_noise_xx.npy'%path, X)
    np.save('%s/data/shift_noise_xy.npy'%path, Y)
    np.save('%s/data/shift_noise_yclass.npy'%path, C)
    print(C,X.shape,C.shape)
    
# =============================================================================
#     for i in range(6):
#         plt.plot(X[i+98],Y[i+98])
# 
#     plt.show()
# =============================================================================
# =============================================================================
#     for i in range(len(X)):
#         plt.plot(X[i], Y[i],linewidth=2.0)
#         plt.show()
# =============================================================================
        
    