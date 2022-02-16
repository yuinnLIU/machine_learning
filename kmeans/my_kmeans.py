# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import *

def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    # dataMat = []              # 文件的最后一个字段是类别标签
    # fr = open(fileName)
    # plot_x, plot_y = [], []
    # for line in fr.readlines():
    #     curLine = line.strip().split('    ')
    #     fltLine = map(float, curLine)    # 将每个元素转成float类型
    #     plot_x.append(fltLine[0])
    #     plot_y.append(fltLine[1])
    #     dataMat.append(fltLine)
    dataset = []
    data = pd.read_csv(fileName, sep='\t')
    x, y = data['x'].tolist(), data['y'].tolist()
    for i in range(len(x)):
        dataset.append([x[i], y[i]])
    return dataset

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataset, k):
    n = 2
    centroids = np.zeros([k, n])   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        dataset = np.array(dataset)
        minJ = min(dataset[:,j])
        maxJ = max(dataset[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataset, k, distMeans =distEclud, createCent = randCent):
    m = len(dataset)
    clusterAssment = np.zeros([m, 2])    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataset, k)

    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataset[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        print(centroids)
        # for cent in range(k):   # 重新计算中心点
        #     # ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
        #     centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment


dataset = loadDataSet('testSet.txt')
# myCentroids,clustAssing = kMeans(dataset, 4)
k = 4
print(random.rand(k, 1))