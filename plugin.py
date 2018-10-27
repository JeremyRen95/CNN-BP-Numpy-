# -*- coding:utf-8 -*-
"""
@project:面试题
@author:JEREMY REN
@file:plugin.py
@ide:PyCharm
@time:2018-10-20 10:36:39
@annotation:define 3 pyfile:1.construct label matrix 2.use matplot draw chart 3.random initial
"""
import numpy as np
import matplotlib.pyplot as plt
import random

def MakeLabel(Y,NumOut):
    num = Y.shape[0]
    LabelMat = np.zeros((num,NumOut))
    LabelMat[np.arange(num),Y] = 1
    return LabelMat

def DrawLosses(loss):
    x = np.arange(len(loss))
    plt.figure()
    plt.plot(x,loss)
    plt.xlabel('sample step')
    plt.ylabel('Loss of predict and true')
    plt.show()

def GetBatch(X,Y,BatchSize):
    num = len(X)
    i = random.randint(1,num - BatchSize)
    return X[i:i + BatchSize], Y[i:i + BatchSize]

