'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''

import numpy as np

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('E:\testSet.txt')
    #循环导入文本数据构成列表
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):#数据转换成numpy类型
    #生成特征矩阵
    dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
    #生成标记矩阵并反置
    labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
    #计算dataMatrix的行列
    m, n = np.shape(dataMatrix)
    #设置移动步长
    alpha = 0.001
    #设置最大递归次数500次
    maxCycles = 500
    #初始化系数为1*3的元素全为1的矩阵
    weights = np.ones((n, 1))
    #循环迭代梯度上升算法
    for k in range(maxCycles):              #heavy on matrix operations
        #计算真实类别与预测类别的差值
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        #调整回归系数
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def testgradascent():
    datamat, labelmat = loadDataSet()
    weights = gradAscent(datamat, labelmat)
    print(weights)

if __name__ == '__mian__':
         testgradascent()
#画出数据集和logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    #把特征集换成数组
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
     
    #循环数据集分类
    xcord1 = []
    ycord1 = []
    xcord2 = [] 
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    #假设 sigmoid 函数为0，并且这里的x,y 相当于上述的x1和x2即可得出y的公式
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        #使用sum函数得出一个值，只用计算一次
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进的随机梯度上升算法，默认迭代150次
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #每次迭代减小alpha的值，但最小为0.01，确保新数据依然有影响。缓解系数波动的情况
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            #选取随机值进行更新
            randIndex = int(np.random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            #删除更新后的值 
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        