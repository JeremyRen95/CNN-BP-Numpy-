# -*- coding:utf-8 -*-
"""
@project:面试题
@author:JEREMY REN
@file:train.py
@ide:PyCharm
@time:2018-10-20 12:52:18
@annotation:training data and construct model
"""
import numpy as np
import pickle
import inference,plugin,layerpart,network,optimizer,lossfunction
from test_set import ShowAcc

#get train data and test data, and normalization respectively
TrainData, TrainLabel, TestData, TestLabel = inference.load()
TrainData,TestData = TrainData/float(255), TestData/float(255)
TrainData -= np.mean(TrainData)
TestData -= np.mean(TestData)

BatchSize = 64 #the size of batch
DimInputData = 784 #the dimension of InputData
DimOutputData = 10 #the dimension of OutputData

print("The batch size of train: " + str(BatchSize) + ", the dimension of InputData: " + str(DimInputData) + ", the dimension of OutputData: " + str(DimOutputData))

### Normal Three Layer Neural Network Test ###
#construct the model of nn
FirstLayerNum = 500 #the first layer of nn has 500 neuron
SecondLayerNum = 200 #the second layer of nn has 200 neuron
NetWorkModel = network.ThreeLayerNet(BatchSize,DimInputData,FirstLayerNum,SecondLayerNum,DimOutputData)
#set the training part of nn
Losses = []
SGDOptimizer = optimizer.SGDMomentum(NetWorkModel.get_params(),lr=0.0001,momentum=0.80,reg=0.00003)
LossFunction = lossfunction.CrossEntropyLoss()

#the train process of nn
IterationNum = 25000
for i in range(IterationNum + 1):
    TrainBatch,LabelBatch = plugin.GetBatch(TrainData,TrainLabel,BatchSize)
    LabelBatch = plugin.MakeLabel(LabelBatch,DimOutputData)

    LabelPred = NetWorkModel.forward(TrainBatch)
    Loss, Dout = LossFunction.get(LabelPred , LabelBatch)
    NetWorkModel.backward(Dout)
    SGDOptimizer.step()

    if i%100 == 0:
        print("Current Training Progress: %s%%, the Number of Iteration of: %s, Current the Value of Loss: %s" % (100*i/IterationNum, i, Loss))
        Losses.append(Loss)

#Export the parameters of NN model
Parameters = NetWorkModel.get_params()
with open("ModelParams.pkl","wb") as f:
    pickle.dump(Parameters,f)

#draw the chart of LossValue
plugin.DrawLosses(Losses)

#the Model Accuracy of TrainData
ShowAcc(NetWorkModel,TrainData,TrainLabel,'Train')
#the Model Accuracy of TestData
ShowAcc(NetWorkModel,TestData,TestLabel,'Test')


