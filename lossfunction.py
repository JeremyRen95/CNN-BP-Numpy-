# -*- coding:utf-8 -*-
"""
@project:面试题
@author:JEREMY REN
@file:lossfunction.py
@ide:PyCharm
@time:2018-10-20 11:33:06
@annotation:Define Loss Function:1.CrossEntropyLoss 2.SoftmaxLoss
"""
import numpy as np
from layerpart import SoftMax

def NLLLoss(YPredic,YTrue):
    loss = 0.0
    num = YPredic.shape[0]
    judge = np.sum(YPredic*YTrue, axis=1)
    for i in judge:
        if(i == 0):
            loss += 500
        else:
            loss += -np.log(i)
    return loss/num

class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, YPredict,YTrue):
        num = YPredict.shape[0]
        softmax = SoftMax()
        Prob = softmax._forward(YPredict)
        loss = NLLLoss(Prob,YTrue)
        YSerial = np.argmax(YTrue, axis=1)
        dout = Prob.copy()
        dout[np.arange(num),YSerial] -= 1
        return loss, dout

class SoftmaxLoss():
    def __init__(self):
        pass

    def get(self,YPredict,YTrue):
        num = YPredict.shape[0]
        loss = NLLLoss(YPredict,YTrue)
        YSerial = np.argmax(YTrue,axis=1)
        dout = YPredict.copy()
        dout[np.arange(num),YSerial] -= 1
        return loss,dout