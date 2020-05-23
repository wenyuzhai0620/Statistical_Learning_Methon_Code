#coding = UTF-8
#Author: Wenyu Zhai
#Date 2020-05-23
#Email: w2zhai@163.com

'''
dataset: Mnist
train_set: 60000
test_set: 10000
----
运行结果：
正确率：81.72%
运行时长：114ss
'''

import numpy as np
import time

def loadData(filename):
    print('start to read data')
    data = []
    label = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        if int(curline[0]) >= 5:
            label.append(1)
        else:
            label.append(-1)
        data.append([int(num)/255 for num in curline[1:]])
    return data, label

def perceptron(data, label, iter=50):
    print('start to trans')
    dataMat = np.mat(data)
    labelMat = np.mat(label).T

    m, n = np.shape(dataMat)
    w = np.zeros((1, np.shape(dataMat)[1]))
    b = 0 
    h = 0.0001 

    for k in range(iter):
        #对每一个样本进行梯度下降
        for i in range(m):
            xi = dataMat[i] #获取当前样本的向量
            yi = labelMat[i] #获取当前向量的标签
            #判断是否为误分类样本
            if -1 * yi * (w * xi.T + b) >= 0:
                #对误分类样本，进行梯度下降，更新w和b
                w = w + h * yi * xi
                b = b + h * yi
        print('Round %d:%d training' % (k, iter))

    return w, b

def modelTest(data, label, w, b):
    print('start to test')
    dataMat = np.mat(data)
    labelMat = np.mat(label).T

    m, n = np.shape(dataMat)
    errorcnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (w * xi.T + b)
        if result >= 0:
            errorcnt += 1
    
    accRate = 1 - (errorcnt / m)
    return accRate

if __name__ == '__main__':
    start = time.time()

    trainData, trainLabel = loadData('mnist_train.csv')
    testData, testLabel = loadData('mnist_test.csv')

    w, b = perceptron(trainData, trainLabel, iter = 30)

    accRate = modelTest(testData, testLabel, w, b)

    end = time.time()

    print('accuracy rate is:', accRate)
    print('time spend:', end - start)    
    

    