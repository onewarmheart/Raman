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


POINTS=1024
START=0
END=1700
    
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
        #plt.plot(xnew, ynew)
        #plt.legend(loc='upper right')
    #plt.show()

# =============================================================================
#     fi = interpolate.interp1d(x, y, kind = kind, bounds_error=False,
#                               fill_value=np.nan)
#     f.append(fi)
#     ynew = fi(xnew) #计算插值结果
#     plt.plot(x, y,color='blue',linewidth=2.0, label='raw data')
#     plt.plot(xnew, ynew,color='red', linewidth=1.0, linestyle='--', label='interpolated')
#     
#     plt.legend(loc='upper right')
#     plt.show()
# =============================================================================
    y_update=np.array(y_update)
    y_update=np.squeeze(y_update)
    return y_update



if __name__ == '__main__':

    path='/home/zr/AI/rruff'
    dire='ex_un'
    x=np.load('%s/data/%s/balan2/uninter_xx_train.npy'%(path,dire))
    y=np.load('%s/data/%s/balan2/uninter_xy_train.npy'%(path,dire))
    print(x.shape,y.shape)
    #x_=np.load('%s/data/%s/remove_2/uninter_xx_val.npy'%(path,dire))
    #y_=np.load('%s/data/%s/remove_2/uninter_xy_val.npy'%(path,dire))
    #plt.figure()
    #plt.xlim((0, 1700))
    #plt.ylim((0, 450000))
    #plt.xlabel('Raman Shift')
    #plt.ylabel('Intensities')

    xy=interpo(x,y)
    print(xy.shape)
    #xy_val=interpo(x_,y_)
    np.set_printoptions(threshold=np.inf)
    np.save('%s/data/%s/balan2/xy_train.npy'%(path,dire),xy)
    #np.save('%s/data/%s/remove_2/xy_val.npy'%(path,dire),xy_val)
    #pl.xticks(fontsize=20)
    #pl.yticks(fontsize=20)
    #pl.legend(loc = 'lower right')
    #pl.show()

