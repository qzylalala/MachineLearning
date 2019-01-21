import numpy as np
import os
import operator

'''将32 * 32 的img转换为1 * 1024的向量'''
def imgVector(filename):
    f = open(filename)
    vector = np.zeros((1, 1024))
    for i in range(32):
        line = f.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(line[j])
    return vector

'''分类器'''
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


def handwritingClassTest():
    labels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        filename = trainingFileList[i]
        file = filename.split('.')[0]
        value = int(file.split('_')[0])
        labels.append(value)
        trainingMat[i, :] = imgVector('trainingDigits/%s' % filename)
    testFileTest = os.listdir('testDigits')
    error = 0.0
    numOfTest = len(testFileTest)
    for i in range(numOfTest):
        filename = testFileTest[i]
        file = filename.split('.')[0]
        value = int(file.split('_')[0])
        TestMat = imgVector('testDigits/%s' % filename)
        result = classify0(TestMat, trainingMat, labels, 3)
        print("the classifier came back with: %d, the real value is: %d " % (result, value))
        if (result != value):
            error += 1
    print("the total error rate is %f" % float(error/numOfTest))


if __name__ == '__main__':
    handwritingClassTest()