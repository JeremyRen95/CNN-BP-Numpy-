# -*- coding:utf-8 -*-
"""
@project:面试题
@author:JEREMY REN
@file:train_LeNet5.py
@ide:PyCharm
@time:2018-10-20 16:14:59
@annotation: training data and construct LeNet5 model
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
TrainData = TrainData.reshape(TrainData.shape[0],1,28,28)
TestData = TestData.reshape(TestData.shape[0],1,28,28)

BatchSize = 16 #the size of batch
DimOutputData = 10 #the dimension of OutputData

print("The batch size of train: " + str(BatchSize) + ", the dimension of InputData: 28*28" + ", the dimension of OutputData: " + str(DimOutputData))

### Normal Three Layer Neural Network Test ###
#construct the model of nn
FirstLayerNum = 500 #the first layer of nn has 500 neuron
SecondLayerNum = 200 #the second layer of nn has 200 neuron
NetWorkModel = network.LeNet5()
#set the training part of nn
Losses = []
SGDOptimizer = optimizer.SGDMomentum(NetWorkModel.get_params(),lr=0.00003)
LossFunction = lossfunction.SoftmaxLoss()

#the train process of nn
IterationNum = 30000
for i in range(IterationNum + 1):
    TrainBatch,LabelBatch = plugin.GetBatch(TrainData,TrainLabel,BatchSize)
    LabelBatch = plugin.MakeLabel(LabelBatch,DimOutputData)

    LabelPred = NetWorkModel.forward(TrainBatch)
    Loss, Dout = LossFunction.get(LabelPred , LabelBatch)
    NetWorkModel.backward(Dout)
    SGDOptimizer.step()

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
