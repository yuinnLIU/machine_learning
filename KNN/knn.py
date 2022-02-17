import math
import pandas as pd


#
# # 测试样本  唐人街探案": [23, 3, 17, "？片"]
# #下面为求与数据集中所有数据的距离代码：
# x = [23, 3, 17]
# KNN = []
# for key, v in movie_data.items():
#     d = math.sqrt((x[0] - v[0]) ** 2 + (x[1] - v[1]) ** 2 + (x[2] - v[2]) ** 2)
#     KNN.append([key, round(d, 2)])
#
# # 输出所用电影到 唐人街探案的距离
# print(KNN)
#
# #按照距离大小进行递增排序
# KNN.sort(key=lambda dis: dis[1])
#
# #选取距离最小的k个样本，这里取k=5；
# KNN=KNN[:5]
# print(KNN)
#
# #确定前k个样本所在类别出现的频率，并输出出现频率最高的类别
# labels = {"喜剧片":0,"动作片":0,"爱情片":0}
# for s in KNN:
#     label = movie_data[s[0]]
#     labels[label[3]] += 1
# labels =sorted(labels.items(),key=lambda l: l[1],reverse=True)
# print(labels,labels[0][0],sep='\n')

import pandas as pd
k = 5
data = pd.read_csv('movie.txt', sep=' ', header=None)
data_np = data.values.tolist()
distance = {}
x = [2, 3, 17]
for i in range(len(data_np)):
    distance[i] = math.sqrt(pow(data_np[i][1] - x[0], 2) + pow(data_np[i][2] - x[1], 2) + pow(data_np[i][3] - x[2], 2))
# print(distance)
sorted_distance = sorted(distance.items(), key = lambda kv:(kv[1], kv[0]))
# print(sorted_distance[:k])
labels = {"喜剧片":0,"动作片":0,"爱情片":0}
for i in range(k):
    # a = data_np[sorted_distance[i][0]][-1]
    if data_np[sorted_distance[i][0]][-1] == "喜剧片":
        labels["喜剧片"] += 1
    elif data_np[sorted_distance[i][0]][-1] == "动作片":
        labels["动作片"] += 1
    else:
        labels["爱情片"] += 1
sorted_labels = sorted(labels.items(), key = lambda kv:(kv[1]), reverse=True)
print(sorted_labels)
print("该部影片最可能的类型是:", sorted_labels[0][0])


#todo
# 一键统计的方法 而不是for循环
# sorted函数 细看
# array list dataframe tuple 含义&互相转换 元组和列表的区别