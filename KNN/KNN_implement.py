import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator


'''从filename中加载出数据'''
def getData(filename):
    f = open(filename)
    arraylines = f.readlines()
    numberlines = len(arraylines)
    returnMat = np.zeros((numberlines, 3))
    classLabel = []
    index = 0
    for line in arraylines:
        line = line.strip()
        list = line.split('\t')
        returnMat[index, :] = list[0:3]
        classLabel.append(int(list[-1]))
        index = index + 1
    return returnMat, classLabel
'''returnmat代表特征， classlabel代表结果'''


'''数据可视化'''
# data, label = getData('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(data[:, 1], data[:, 2], 15.0*np.array(label), 15.0*np.array(label))
# plt.show()


'''将数据加权，这里就简单的将数据放在[0, 1]之间'''
def autoNorm(data):
    minV = data.min(0)
    maxV = data.max(0)
    ranges = maxV - minV
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minV, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData, ranges, minV


'''分类函数/分类器'''
def classify0(inX,dataSet,labels,k):
    # 计算inX与训练集之间的距离，并排序
    dataSetSize = dataSet.shape[0]  #行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()  #返回索引值
    classCount = {}
    #对前K个的标签进行统计
    for i in range(k):
        votelLabel = labels[sortedDistIndicies[i]]
        classCount[votelLabel] = classCount.get(votelLabel, 0)+1
    # 对统计的标签数量进行降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]


'''测试函数'''
def dataTest():
    per = 0.1
    data, label = getData('datingTestSet2.txt')
    normMat, ranges, minV = autoNorm(data)
    m = normMat.shape[0]
    numOfTest = int(m*per)
    error = 0.0
    for i in range(numOfTest):
        result = classify0(normMat[i, :], normMat[numOfTest:m, :], label[numOfTest:m], 3)
        print('the classifier came back with: %d, the real answer is: %d'%(result, label[i]))
        if (result != label[i]):
            error += 1.0
    print('the error rate is: %f' % (error/float(numOfTest)))


'''最后一步，进行交互'''
def classify():
    resultList = ['not at all', 'in small doses', 'in large doses']
    gameTime = float(input("percentage of time spent playing video games? ->"))
    ffmiles = float(input("frequent flier miles earned per year? ->"))
    iceCream = float(input("liters of ice cream consumed per year? ->"))
    data, label = getData('datingTestSet2.txt')
    normData, ranges, minV = autoNorm(data)
    inArray = np.array([gameTime, ffmiles, iceCream])
    result = classify0((inArray - minV)/ranges, normData, label, 3)
    print("you will probably like this person:", resultList[result - 1])


if __name__ == '__main__':
    classify()
