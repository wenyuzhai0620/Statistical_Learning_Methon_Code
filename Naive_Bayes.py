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
正确率：84.3%
运行时长：272s
'''

import numpy as np
import time

def loadData(filename):
    data = []
    label = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        data.append([int(int(num)>128) for num in curline[1:]])
        label.append(int(curline[0]))
    return data, label

def NaiveBayes(py, px_y, x):
    '''
    py: 先验概率分布
    px_y: 条件概率分布
    x: 要估计的样本x
    return: 返回所有label的估计概率
    '''
    featureNum = 784
    classNum = 10
    P = [0] * classNum
    for i in range(classNum):
        sum = 0
        for j in range(featureNum):
            sum += px_y[i][j][x[j]]
        P[i] = sum + py[i]
    return P.index(max(P))

def model_test(py, px_y, testData, testLabel):
    errorcnt = 0
    for i in range(len(testData)):
        presict = NaiveBayes(py, px_y, testData[i])
        if presict != testLabel[i]:
            errorcnt += 1
    
    return 1 - (errorcnt / len(testData))

def getAllProbability(trainData, trainLabel):
    featureNum = 784
    classNum = 10

    py = np.zeros((classNum, 1))
    for i in range(classNum):
        py[i] = ((np.sum(np.mat(trainLabel) == i)) + 1) / (len(trainLabel) + 10)
    py = np.log(py)

    px_y = np.zeros((classNum, featureNum, 2))
    for i in range(len(trainLabel)):
        label = trainLabel[i]
        x = trainData[i]
        for j in range(featureNum):
            px_y[label][j][x[j]]  += 1

    for label in range(classNum):
         for j in range(featureNum):
             px_y0 = px_y[label][j][0]
             px_y1 = px_y[label][j][1]
             px_y[label][j][0] = np.log((px_y0+1) / (px_y0 + px_y1 + 2))
             px_y[label][j][1] = np.log((px_y1+1) / (px_y0 + px_y1 + 2))

    return py, px_y

if __name__ == "__main__":
    start = time.time()
    print('start read transSet')
    trainData, trainLabel = loadData('mnist_train.csv')
    print('start read testSet')
    testData, testLabel = loadData('mnist_test.csv')

    print('start to train')
    py, px_y = getAllProbability(trainData, trainLabel)

    print('start to test')
    accuracy = model_test(py, px_y, testData, testLabel)

    print('the accuracy is: ', accuracy)
    print('time spend:', time.time() - start)