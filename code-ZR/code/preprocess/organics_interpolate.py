# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:23:58 2018

@author: ZR
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


POINTS=1100
START=200
END=3700
    
def interpo(x,y,kind = 'linear'):    
    #load数据集  
    xnew = np.linspace(START, END, POINTS)
    y_update=[]
    f=[]
    for i in range(len(y)):
        y_update.append([])
        fi = interpolate.interp1d(x[i], y[i], kind = kind, bounds_error=False,
                                  fill_value=0)
        f.append(fi)
        ynew = fi(xnew) #计算插值结果
        y_update[i].append(ynew)

    y_update=np.array(y_update)
    y_update=np.squeeze(y_update)
    return y_update



if __name__ == '__main__':


    x=np.load('/home/zr/AI/pretrain/data/shift_noise_xx.npy')
    y=np.load('/home/zr/AI/pretrain/data/shift_noise_xy.npy')

    x_data=interpo(x,y)
    x_data = x_data[:,:1024]
    np.set_printoptions(threshold=10000)
    np.save('/home/zr/AI/pretrain/data/inter_xy.npy',x_data)
    print(x_data.shape)
    
    xnew = np.linspace(START, END, POINTS)
    plt.plot(x[0],y[0],color='blue',linewidth=2.0,label='raw')
    plt.plot(xnew[:1024], x_data[0],color='red',linewidth=1.0,label='interpolate')
    plt.xlabel('Raman Shift')
    plt.ylabel('Intensities')
    plt.legend(loc = 'best')
    plt.show()

    for i in range(len(x)):
        plt.plot(x[i],y[i],label='raw')
        
    #plt.legend(loc='upper right')
    plt.show()
    
    for i in range(len(x_data)):
        plt.plot(xnew[:1024],x_data[i],label='interpolate')
    
    #plt.legend(loc='upper right')
    plt.show()
