import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data = pd.read_csv('xclara.csv')

# 将csv文件中的数据转换为二维数组
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
# # Getting the cluster centers
C = kmeans.cluster_centers_  #中心点坐标

# plt.scatter(f1, f2, c='black', s=5)
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
# 不同的子集使用不同的颜色
k = 3
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=5, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')

plt.show()