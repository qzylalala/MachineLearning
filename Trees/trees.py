from math import log
import operator


'''计算香农熵'''
def calShannon(dataSet):
    numOfData = len(dataSet)
    labelCounts = {}
    for featV in dataSet:
        currentLabel = featV[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannon = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numOfData
        shannon -= prob*log(prob, 2)
    return shannon


'''创建数据集'''
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


'''按特征划分数据集,把符合要求的元素取出来'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featV in dataSet:
        if featV[axis] == value:
            reduceFeatV = featV[:axis]
            reduceFeatV.extend(featV[axis+1:])
            retDataSet.append(reduceFeatV)
    return retDataSet


'''选择最佳划分方案'''
def choose(dataSet):
    numOfFeat = len(dataSet[0]) - 1
    shannon = calShannon(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numOfFeat):
        featList = [example[i] for example in dataSet]
        uniqueV = set(featList)
        newShannon = 0.0
        for value in uniqueV:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newShannon += prob * calShannon(subDataSet)
        infoGain = shannon - newShannon
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


'''决定子节点分类'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedCount[0][0]


'''创建树的函数'''
def createTrees(dataSet, labels):
    classList  =[example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0])  == 1:
        return majorityCnt(classList)
    bestFeat = choose(dataSet)
    bestLabel = labels[bestFeat]
    myTree = {bestLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestLabel][value] = createTrees(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


myData, labels = createDataSet()
myTree = createTrees(myData, labels)
print(myTree)