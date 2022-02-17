
#导入需要的库
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
torch.manual_seed(10)

#1.生成数据
# sample_nums = 100
# mean_value = 1.7
# bias = 1
# n_data = torch.ones(sample_nums, 2) #100行 2列
#
# #torch.normal(mean, std, *, generator=None, out=None) 均值 标准差
# x0 = torch.normal(mean_value * n_data, 1) + bias      # 类别0 数据 shape=(100, 2) 均值为1.7 标准差为1
# y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100, 1) 标签为0
#
# x1 = torch.normal(-mean_value * n_data, 1) + bias     # 类别1 数据 shape=(100, 2) 均值为-1.7 标准差为1
# y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100, 1) 标签为1
#
# train_x = torch.cat((x0, x1), 0)  #共有200个训练样本
# train_y = torch.cat((y0, y1), 0)  #200个训练样本对应的类别

y = pd.read_csv('iris.data', header=None)
x = y.iloc[0:100, [0, 2]].values  # 取前100列 只包含两类 做二分类 只取前两个特征
y = y.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)  # np.where(condition, x, y)

train_x = torch.Tensor(x)
train_y = torch.Tensor(y)

#2.定义模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.features = nn.Linear(2, 1) #全连接层 w0 w1 b
        self.sigmoid = nn.Sigmoid() #激活函数

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

#实例化
model = LogisticRegression()
print(model.features)

#3.模型训练
lr = 0.1
epoch = 100
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) #优化器
loss_func = nn.BCELoss()

for i in range(epoch):
    y_pred = model(train_x) #y的预测值
    y_pred = y_pred.squeeze()
    loss = loss_func(y_pred, train_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() #梯度清零
    # print(train_y, y_pred)

    #绘制图像
    if i % 20 == 0:
        res = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类 .ge #greater and equal >=0.5 取值为1 反之为0
        correct = (res == train_y).sum()  # 统计正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率
        print(acc)

        plt.scatter(x[0:50, 0], x[0:50, 1], c='r', label='class 0')
        plt.scatter(x[50:100, 0], x[50:100, 1], c='b', label='class 1')

        #
        w0, w1 = model.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        b = float(model.features.bias[0].item())
        # # print(w0, w1, b)
        #
        plot_x = np.arange(4, 8, 0.1) #自变量取值范围
        plot_y = (-w0 * plot_x - b) / w1 #因变量 w0x + w1y + b = 0

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color': 'red'})

        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(i, w0, w1, b, acc))

        plt.plot(plot_x, plot_y)
        # plt.savefig('Iteration{}.png'.format(i))
        plt.show()

