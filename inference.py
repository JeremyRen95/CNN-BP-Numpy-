# -*- coding:utf-8 -*-
"""
@project:面试题
@author:JEREMY REN
@file:inference.py
@ide:PyCharm
@time:2018-10-20 09:53:09
@annotation:load mnist file and stored as a pickle file
"""
import numpy as np
from urllib import request
import gzip
import pickle

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def DownloadMnist():
    MnistUrl = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading"+name[1]+"...")
        request.urlretrieve(MnistUrl+name[1],name[1])
    print("DownLoad is complete!")

def SaveMnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1],'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    DownloadMnist()
    SaveMnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()