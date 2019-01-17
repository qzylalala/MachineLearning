import pandas as pd
import numpy as np
import operator


'''获取csv文件中的数据，data为features， labels为labels'''
'''选取了pclass， sex(1是男， 2是女), age, sibsp, Fare, parch共6个Features,返回值形式为矩阵'''
def getData():
    data = pd.read_csv('train.csv')
    useful = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
    useful.dropna(axis=0, how='any', inplace=True)
    labels = useful['Survived']
    data = useful.drop(columns=['Survived'])
    data = data.values
    for i in data:
        if i[1] == 'male':
            i[1] = 1
        else:
            i[1] = 0
    labels = labels.values
    labels = labels.reshape(714, 1)
    data = data.astype(float)
    return data, labels


def classfyByFeat(inX, data, labels):
    datasize = data.shape[0]
    diffMat = np.tile(inX, (datasize, 1)) - data
    sqDiffMat = diffMat**2
    sqSum = sqDiffMat.sum(axis=1)
    distances = sqSum**0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(3):
        votelLabel = labels[sortedDistances[i]][0]
        if votelLabel not in classCount:
            classCount[votelLabel] = 0
        classCount[votelLabel] += 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return labels[maxIndex]

def autoNorm(data):
    minV = data.min(0)
    maxV = data.max(0)
    ranges = maxV - minV
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minV, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


def predict():
    resultList = []
    data, labels = getData()
    normdata = autoNorm(data)
    testData = pd.read_csv('test.csv')
    useful = testData.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
    useful = useful.fillna(axis=0, method='ffill')
    testData = useful.values
    for i in testData:
        if i[1] == 'male':
            i[1] = 1
        else:
            i[1] = 0
    testData = testData.astype(float)
    testNormdata = autoNorm(testData)
    num = 892
    Id = []
    for i in testNormdata:
        result = classfyByFeat(i, normdata, labels)
        resultList.append(result[0])
        Id.append(num)
        num = num + 1
    return resultList, Id


if __name__ == '__main__':
    result, id = predict()
    dataframe = pd.DataFrame({'PassengerId':id, 'Survived':result})
    dataframe.to_csv('result.csv', index=False, sep=',')
