# -*- coding:utf-8 -*-
"""
@project:面试题
@author:JEREMY REN
@file:network.py
@ide:PyCharm
@time:2018-10-20 10:33:34
@annotation: Constructing neural network
"""
import numpy as np
import plugin
from layerpart import FullConnectLayer, ReLU, SoftMax, Dropout, Conv, MaxPool
import pickle
from abc import ABCMeta, abstractmethod

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

class ThreeLayerNet(Net):
    """
    Construct 3 layer NN
    """
    def __init__(self,BatchSize,NumIn,N1,N2,NumOut,Weights=''):
        self.FirstLayer = FullConnectLayer(NumIn,N1)
        self.FirstLayerActivation = ReLU()
        self.SecondLayer = FullConnectLayer(N1,N2)
        self.SecondLayerActivation = ReLU()
        self.ThirdLayer = FullConnectLayer(N2,NumOut)

        if Weights == '':
            pass
        else:
            with open(Weights,'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        output1 = self.FirstLayer._forward(X)
        actoutput1 = self.FirstLayerActivation._forward(output1)
        output2 = self.SecondLayer._forward(actoutput1)
        actoutput2 = self.SecondLayerActivation._forward(output2)
        output3 = self.ThirdLayer._forward(actoutput2)
        return output3

    def backward(self, dout):
        dout = self.ThirdLayer._backward(dout)
        dout = self.SecondLayerActivation._backward(dout)
        dout = self.SecondLayer._backward(dout)
        dout = self.FirstLayerActivation._backward(dout)
        dout = self.FirstLayer._backward(dout)

    def get_params(self):
        return [self.FirstLayer.W, self.FirstLayer.b, self.SecondLayer.W, self.SecondLayer.b, self.ThirdLayer.W, self.ThirdLayer.b]

    def set_params(self, params):
        [self.FirstLayer.W, self.FirstLayer.b, self.SecondLayer.W, self.SecondLayer.b, self.ThirdLayer.W, self.ThirdLayer.b] = params

class LeNet5(Net):
    """
    Construct Convolutional neural network(LeNet5)
    """
    def __init__(self):
        self.conv1 = Conv(1, 6, 5)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2, 2)
        self.conv2 = Conv(6, 16, 5)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2, 2)
        self.FirstLayer = FullConnectLayer(16 * 4 * 4, 120)
        self.ReLU3 = ReLU()
        self.SecondLayer = FullConnectLayer(120, 84)
        self.ReLU4 = ReLU()
        self.ThirdLayer = FullConnectLayer(84, 10)
        self.Softmax = SoftMax()

        self.p2_shape = None

    def forward(self, X):
        Conv1Output = self.conv1._forward(X)
        actConv1Output = self.ReLU1._forward(Conv1Output)
        Poollayer1 = self.pool1._forward(actConv1Output)
        Conv2Output = self.conv2._forward(Poollayer1)
        actConv2Output = self.ReLU2._forward(Conv2Output)
        Poollayer2 = self.pool2._forward(actConv2Output)
        self.p2_shape = Poollayer2.shape
        fl = Poollayer2.reshape(X.shape[0], -1)  # Flatten convert to one column matrix
        Output3FullConnection = self.FirstLayer._forward(fl)
        actOutput3 = self.ReLU3._forward(Output3FullConnection)
        Output4FullConnection = self.SecondLayer._forward(actOutput3)
        actOutput4 = self.ReLU4._forward( Output4FullConnection)
        Output5FullConnection = self.ThirdLayer._forward(actOutput4)
        actOutput5 = self.Softmax._forward(Output5FullConnection)
        return actOutput5

    def backward(self, dout):
        dout = self.ThirdLayer._backward(dout)
        dout = self.ReLU4._backward(dout)
        dout = self.SecondLayer._backward(dout)
        dout = self.ReLU3._backward(dout)
        dout = self.FirstLayer._backward(dout)
        dout = dout.reshape(self.p2_shape)
        dout = self.pool2._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FirstLayer.W, self.FirstLayer.b, self.SecondLayer.W, self.SecondLayer.b, self.ThirdLayer.W, self.ThirdLayer.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FirstLayer.W, self.FirstLayer.b, self.SecondLayer.W, self.SecondLayer.b, self.ThirdLayer.W, self.ThirdLayer.b] = params