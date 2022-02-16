# coding: utf-8
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def dist(a, b, ax=1): # 距离函数
    return np.linalg.norm(a - b, axis=ax) # 求范数公式 默认二范数

# 读取数据
def loaddata(datapath):
    data = pd.read_csv(datapath)
    f1 = data['V1'].values
    f2 = data['V2'].values
    X = np.array(list(zip(f1, f2)))
    # 绘制原始数据
    plt.scatter(f1, f2, s=5, c='black')
    plt.title('original distribution')
    return X

def kmeans(datapath, k):
    dataset = loaddata(datapath)

    #随机初始化中心点
    C_x = np.random.randint(0, np.max(dataset) - 20, size=k)
    C_y = np.random.randint(0, np.max(dataset) - 20, size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    fig, ax = plt.subplots()
    plt.scatter(np.transpose(dataset)[0], np.transpose(dataset)[1], s=5, c='black')
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')
    plt.title('random center')


    # 用于保存中心点更新前的坐标
    C_old = np.zeros(C.shape)

    # 用于保存数据所属中心点
    clusters = np.zeros(len(dataset))

    # 计算新旧中心点的距离
    error = dist(C, C_old, None)

    count = 1
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(dataset)):
            distances = dist(dataset[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [dataset[j] for j in range(len(dataset)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)

        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        fig, ax = plt.subplots()
        # 不同的子集使用不同的颜色
        for i in range(k):
            points = np.array([dataset[j] for j in range(len(dataset)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=5, c=colors[i])
        ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')
        # plt.title('iteration ', count)
        plt.title("Iteration: {}\n".format(count))
        plt.savefig('Iteration{}.png'.format(count))
        count += 1

    return C[:, 0], C[:, 1]


data_dir = 'xclara.csv'
center_x, center_y = kmeans(data_dir, 3)
# plt.show()
