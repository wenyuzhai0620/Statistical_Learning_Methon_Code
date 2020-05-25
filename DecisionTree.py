#coding = UTF-8
#Author: Wenyu Zhai
#Date 2020-05-25
#Email: w2zhai@163.com

'''
dataset: Mnist
train_set: 60000
test_set: 10000
----
运行结果：ID3(未剪枝)
正确率：8%
运行时长：
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
    
def majorClass(label):
    '''
    找到当前标签集中占数目最大的标签
    label: 标签集
    return: 最大的标签
    '''
    classDict = {} # build dictionary, using different categories label
    for i in range(len(label)):
        if label[i] in classDict.keys():
            classDict[label[i]] += 1
        else:
            classDict[label[i]] = 1
    # sort dictionary decending
    classSort = sorted(classDict.items(), key = lambda x: x[1], reverse = True)    
    return classSort[0][0]

def calc_H_D(trainLabel):
    # calculate Empirical Entropy of dataset D
    h_d = 0
    trainLabelSet = set([label for label in trainLabel])
    for i in trainLabelSet:
        p = trainLabel[trainLabel == i].size / trainLabel.size
        h_d += -1 * p * np.log2(p)
    return h_d

def calcH_D_A(trainData_DevFeature, trainLabel):
    '''
    calculate Empirical Conditional Entropy
    trainData_DevFeature: 切割后只有feature那列数据的数组
    '''
    h_d_a = 0
    trainDataSet = set([label for label in trainData_DevFeature])
    for i in trainDataSet:
        h_d_a += trainData_DevFeature[trainData_DevFeature == i].size / trainData_DevFeature.size \
                * calc_H_D(trainLabel[trainLabel == 1])
    return h_d_a

def calcBestFeature(trainDataList, trainLabelList):
    trainData = np.array(trainDataList)
    trainLabel = np.array(trainLabelList)

    featureNum = trainData.shape[1]
    maxg_d_a = -1
    maxFeature = -1

    h_d = calc_H_D(trainLabel)
    for feature in range(featureNum):
        trainData_DevideByFeature = np.array(trainData[:, feature].flat)
        g_d_a = h_d - calcH_D_A(trainData_DevideByFeature, trainLabel)
        if g_d_a > maxg_d_a:
            maxg_d_a = g_d_a
            maxFeature = feature
    return maxFeature, maxg_d_a

def getSubData(trainData, trainLabel, A, a):
    '''
    update dataset and labelset
    A: 要去除的特征索引
    a: 当data[A] == a时，说明该行样本是要保留的
    return: 新的数据集和标签集
    '''
    retData = []
    retLabel = []
    for i in range(len(trainData)):
        if trainData[i][A] == a:
            retData.append(trainData[i][0:A] + trainData[i][A+1:])
            retLabel.append(trainLabel[i])
    return retData, retLabel

def createTree(*dataset):
    '''
    递归创建决策树
    dataset: (trainDataList, trainLabelList) <<-- 元祖形式
    return: 新的子节点或该叶子节点的值
    '''
    Epsilon = 0.1
    trainDataList = dataset[0][0]
    trainLabelList = dataset[0][1]
    print('start a node', len(trainDataList[0]), len(trainLabelList))

    classDict = {i for i in trainLabelList}
    if len(classDict) == 1:
        return trainLabelList[0]
    if len(trainDataList[0]) == 0:
        return majorClass(trainLabelList)

    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)    

    treeDict = {Ag:{}}
    treeDict[Ag][0] = createTree(getSubData(trainDataList, trainLabelList, Ag, 0))
    treeDict[Ag][1] = createTree(getSubData(trainDataList, trainLabelList, Ag, 1))
    return treeDict

def predict(testDataList, tree):
    while True:
        (key, value), = tree.items()
        if type(tree[key]).__name__ == 'dict':
            dataval = testDataList[key]
            del testDataList[key]
            tree = value[dataval]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value

def model_test(testDataList, testLabelList, tree):
    errorcnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(testDataList[i], tree):
            errorcnt += 1
    return 1 - errorcnt / len(testDataList)


if __name__ == '__main__':
    start = time.time()
    trainData, trainLabel = loadData('mnist_train.csv')
    testData, testLabel = loadData('mnist_test.csv')
    
    print('start to crate tree')
    tree = createTree((trainData, trainLabel))
    print('tree is:', tree)

    print('start test')
    acc = model_test(testData, testLabel, tree)
    print('the accuracy is:', acc)

    end = time.time()
    print('time spend is: ', end - start)
