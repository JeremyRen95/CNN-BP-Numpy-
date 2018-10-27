# -*- coding:utf-8 -*-
"""
@project:面试题
@author:JEREMY REN
@file:test_set.py
@ide:PyCharm
@time:2018-10-20 14:38:31
@annotation: Show the Accuracy of Model
"""
import numpy as np

def ShowAcc(Model,Data,Label,type):
    LabelPred = Model.forward(Data)
    result = np.argmax(LabelPred,axis=1) - Label
    result = list(result)
    print(type + "Correct " +str(result.count(0)) + " of " + str(Data.shape[0]) + ", accuracy: " + str(100*result.count(0)/Data.shape[0]) + "%")


