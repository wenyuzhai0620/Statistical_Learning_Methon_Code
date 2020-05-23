#coding = UTF-8
#Author: Wenyu Zhai
#Date 2020-05-23
#Email: w2zhai@163.com

'''
dataset: Mnist
train_set: 60000
test_set: 10000
----
运行结果：(k近邻：25)
欧式距离
    acc: 97% 
    time: 483s
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
        data.append([int(num) for num in curline[1:]])
        label.append(int(curline[0]))
    return data, label

def calcDist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))
    #Manhattan distence
    #return np.sum(x1 - x2)

def getClosest(trainDataMat, trainLabelMat, x, topK):
    distlist = [0] * len(trainDataMat)
    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = calcDist(x1, x)
        distlist[i] = curDist

    topKList = np.argsort(np.array(distlist))[:topK]
    labelList = [0] * 10
    for index in topKList:
        labelList[int(trainLabelMat[index])] += 1

    return labelList.index(max(labelList))

def model_test(trainData, trainLabel, testData, testLabel, topK):
    print('start test')
    trainDataMat = np.mat(trainData)
    trainLabelMat = np.mat(trainLabel).T
    testDataMat = np.mat(testData)
    testLabelMat = np.mat(testLabel).T

    errorcnt = 0
    for i in range(200):
        print('test %d:%d' % (i, 200))
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        if y != testLabelMat[i]: errorcnt += 1

    return 1 - (errorcnt / 200)

if __name__ == '__main__':
    start = time.time()

    trainData, trainLabel = loadData('mnist_train.csv')
    testData, testLabel = loadData('mnist_test.csv')

    acc = model_test(trainData, trainLabel, testData, testLabel, 25)
    print('accuracy is %d' %(acc * 100), '%')

    end = time.time()

    print('time spend:', end - start)
