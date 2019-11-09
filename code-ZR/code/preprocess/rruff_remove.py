# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:11:19 2018

@author: ZR
"""

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

#path='E:/MLXMU3/pred/rruff'
path='/home/zr/AI/rruff'
dire='ex_un'

#只保留含4个及以上数据的类别
def dele(X,Y,C,N):
    
    X_pre=[]
    Y_pre=[]
    C_pre=[]
    N_pre=[]
    x_aug=[]
    y_aug=[]
    c_aug=[]
    n_aug=[]
    c=0
    num=0
    for i in range(len(C)):
        if C[i]==c:
            x_aug.append(X[i])
            y_aug.append(Y[i])
            c_aug.append(num)
            n_aug.append(N[i])
        else:
            if len(c_aug)>=3:
                if len(c_aug)<=20:
                    for m in range(len(c_aug)):
                        X_pre.append(x_aug[m])
                        Y_pre.append(y_aug[m])
                        C_pre.append(c_aug[m])
                        N_pre.append(n_aug[m])
                    num+=1
                else:
                    for m in range(20):
                        X_pre.append(x_aug[m])
                        Y_pre.append(y_aug[m])
                        C_pre.append(c_aug[m])
                        N_pre.append(n_aug[m])
                    num+=1
            x_aug=[X[i]]
            y_aug=[Y[i]]
            c_aug=[num]
            n_aug=[N[i]]
            c=C[i]
            
    return np.array(X_pre),np.array(Y_pre),np.array(C_pre),np.array(N_pre)

if __name__ == '__main__':
    
    data_x = np.load('%s/data/%s/xx_raw.npy'%(path,dire))
    data_y = np.load('%s/data/%s/xy_raw.npy'%(path,dire))
    data_c = np.load('%s/data/%s/yclass.npy'%(path,dire))
    name   = np.load('%s/data/%s/realname.npy'%(path,dire))
    print(data_x.shape,name.shape)
    x,y,c,n = dele(data_x,data_y,data_c,name)
    
    np.save('%s/data/%s/remove_2/xx.npy'%(path,dire), x)
    np.save('%s/data/%s/remove_2/xy.npy'%(path,dire), y)
    np.save('%s/data/%s/remove_2/yclass.npy'%(path,dire),c)
    np.save('%s/data/%s/remove_2/realname.npy'%(path,dire),n)
    print(c.shape,n.shape)
    #np.set_printoptions(threshold=100)
    #print(C)
    #print(data_x.shape)
    #print(c.shape)
    #print('before:',data_c)
    #print('removed:', c)
          