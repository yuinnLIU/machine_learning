import numpy as np

x = np.array([[-1, -1, 0, 2, 0],
              [-2, 0, 0, 1, 1]])
print(x)
# 以x为例，用PCA方法将这两行数据降到一行

# step1: 中心化
# 因为x矩阵的每行已经是零均值，此步略过

# step2：求协方差矩阵
# C=1/n(XtX)
x_t = np.transpose(x)
print(x_t)
C = 1/5 * np.dot(x, x_t)
# print(C)

# C = np.cov(x)

# step3: 计算矩阵的特征值和特征向量
b = np.linalg.eig(C)
print(b)
# (array([2. , 0.4]),
#  array([[ 0.70710678, -0.70710678],
#        [ 0.70710678,  0.70710678]]))

# pc1 = 较大特征值对应的的特征向量
pc1 = np.transpose(b[1])[0]

# step4: 计算降维后的x值
res_x = np.dot(x_t, pc1)
print(res_x)