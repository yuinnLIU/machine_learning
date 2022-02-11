#导入需要的库
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

#1.生成数据
sample_nums = 100
train_x = torch.linspace(0,1,sample_nums)  #0-1 均匀生成100个点
# print(train_x.shape)
train_x = torch.unsqueeze(train_x,dim=1)  #将数据变成一列
# print(train_x)
k = 2
train_y = k * train_x + torch.rand(train_x.size())  #torch.rand() 随机生成0-1的数

#2.定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.Linear = torch.nn.Linear(1, 1) #全连接层 k b

    def forward(self, x):
        x = self.Linear(x)
        return x

#实例化
# model = LogisticRegression()
model = LinearRegression()
# print(model.features)


#3.模型训练
lr = 0.1
epoch = 100
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) #优化器

loss_func = nn.MSELoss()

for i in range(epoch):
    y_pred = model(train_x) #y的预测值

    loss = loss_func(y_pred, train_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() #梯度清零



    #绘制图像
    if i % 20 == 0:
        plt.scatter(train_x, train_y, c='r')
        plot_y = y_pred.detach().numpy()
        plt.plot(train_x.numpy(), plot_y)

        [pred_k, pred_b] = model.parameters()
        # print(pred_k, pred_b)
        pred_y = pred_k * train_x + pred_b
        pred_y = pred_y.detach().numpy()
        # print(pred_y)
        plt.plot(train_x.numpy(), pred_y, c='y')

        plt.title("Iteration: {}\n loss:{:.2}".format(i, loss))
        # plt.savefig('Iteration{}.png'.format(i))
        plt.show()




