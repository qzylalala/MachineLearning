import pandas as pd
import numpy as np


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
    addition = np.ones(714)
    data = np.insert(data, 0, values=addition, axis=1)
    labels = labels.values
    labels = labels.reshape(714, 1)
    data = data.astype(float)
    return data, labels

'''获取theta，相当于训练完成'''
def getTheta():
    data, labels = getData()
    first = np.dot(data.T, data)
    second = np.linalg.inv(first)
    third = np.dot(second, data.T)
    fina = np.dot(third, labels)
    return fina.flatten()


def Test():
    result = []
    ID = []
    data = pd.read_csv('test.csv')
    useful = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
    useful = useful.fillna(axis=0, method='ffill')
    data = useful.values
    addition = np.ones(418)
    data = np.insert(data, 0, values=addition, axis=1)
    for i in data:
        if i[2] == 'male':
            i[2] = 1
        else:
            i[2] = 0
    data = data.astype(float)
    num = 892
    for i in data:
        theta = getTheta()
        prob = np.dot(i, theta.transpose())
        if abs(prob) > 0.5:
            result.append(1)
        else:
            result.append(0)
        ID.append(num)
        num += 1
    return result, ID


if __name__ == '__main__':
    result, ID = Test()
    dataframe = pd.DataFrame({'PassengerId':ID, 'Survived':result})
    dataframe.to_csv('result.csv', index=False, sep=',')