import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def load_data(datapath, plot = True):
    y = pd.read_csv(datapath, header=None)
    x = y.iloc[0:100, [0, 2]].values  # 取前100列 只包含两类 做二分类 只取前两个特征
    y = y.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)  # np.where(condition, x, y)

    if plot:
        plt.scatter(x[:50, 0], x[:50, 1], color='red')
        plt.scatter(x[50:100, 0], x[50:100, 1], color='blue')
        plt.show()
    return x, y

class Perceptron(nn.Module):
  def __init__(self):
    super(Perceptron, self).__init__()
    self.l1 = nn.Linear(2, 2)
    self.l2 = nn.Linear(2, 2)

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    return F.sigmoid(x)


model = Perceptron()
lr = 0.1
epoch = 100
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) #优化器
loss_func = nn.MSELoss()


x, y = load_data('iris.data', False)
train_x, train_y = torch.Tensor(x), torch.Tensor(y)

for i in range(epoch):
    y_pred = model(train_x) #y的预测值
    # y_pred = y_pred.squeeze()
    y_pred = torch.argmax(y_pred, -1)
    loss = loss_func(y_pred, train_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() #梯度清零
    # print(train_y, y_pred)

    #绘制图像
    if i % 20 == 0:
        # res = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类 .ge #greater and equal >=0.5 取值为1 反之为0
        correct = (y_pred == train_y).sum()  # 统计正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率
        print(acc)

        plt.scatter(train_x.data.numpy()[0:50, 0], train_x.data.numpy()[0:50, 1], c='r', label='class 0')
        plt.scatter(train_x.data.numpy()[50:100, 0], train_x.data.numpy()[50:100, 1], c='b', label='class 0')


        # w0, w1 = model.features.weight[0]
        # w0, w1 = float(w0.item()), float(w1.item())
        # b = float(model.features.bias[0].item())
        # print(w0, w1, b)
        #
        # plot_x = np.arange(-6, 6, 0.1) #自变量取值范围
        # plot_y = (-w0 * plot_x - b) / w1 #因变量 w0x + w1y + b = 0
        #
        # plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color': 'red'})
        #
        # plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(i, w0, w1, b, acc))
        #
        # plt.plot(plot_x, plot_y)
        # plt.savefig('Iteration{}.png'.format(i))
        plt.show()




