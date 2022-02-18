import torch
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# 数据集的预处理
data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]
)

data_path = 'dataset'
# 获取数据集
train_data = mnist.MNIST(data_path,train=True,transform=data_tf,download=False)
test_data = mnist.MNIST(data_path,train=False,transform=data_tf,download=False)

# 获取迭代数据：data.DataLoader()
train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True)
test_loader = data.DataLoader(test_data,batch_size=1,shuffle=True)

# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2, #尺寸减半
                            padding=1),
            torch.nn.BatchNorm2d(16), #参数为通道数
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64,100)
        self.mlp2 = torch.nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x) #(16,14,14)
        x = self.conv2(x) #(32,7,7)
        x = self.conv3(x) #(64,4,4)
        x = self.conv4(x) #(64,2,2)
        # print(x.view(x.size(0), -1).shape)
        x = self.mlp1(x.view(x.size(0),-1)) #(100)
        x = self.mlp2(x) #(10)
        return x

model = CNNnet()
print(model)

# 定义损失函数和优化器
loss_func = torch.nn.CrossEntropyLoss() # 交叉熵损失
opt = torch.optim.Adam(model.parameters(),lr=0.001) #Adam优化器

# 训练网络
loss_count = []
def train():
    for epoch in range(2):
        for i,(x,y) in enumerate(train_loader):
            batch_x = Variable(x) # torch.Size([128, 1, 28, 28])  batchsize=128 注：tensor不能反向传播，variable可以反向传播。
            batch_y = Variable(y) # torch.Size([128])
            # 获取最后输出
            out = model(batch_x) # torch.Size([128,10])
            # 获取损失
            loss = loss_func(out,batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            if i%20 == 0:
                loss_count.append(loss)
                print('{}:\t'.format(i), loss.item())
                torch.save(model,'model')
            if i % 100 == 0:
                for a,b in test_loader:
                    test_x = Variable(a)
                    test_y = Variable(b)
                    out = model(test_x)
                    # print('test_out:\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
                    print('accuracy:\t',accuracy.mean())
                    break
    plt.figure('PyTorch_CNN_Loss')
    plt.plot(loss_count,label='Loss')
    plt.legend()
    plt.show()

def test():
# 测试网络
    model = torch.load('model')
    accuracy_sum = []
    for i,(test_x,test_y) in enumerate(test_loader):
        test_x = Variable(test_x)
        test_y = Variable(test_y)
        out = model(test_x)
        accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
        accuracy_sum.append(accuracy.mean())
        print('accuracy:\t',accuracy.mean())

    print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
    # 精确率图
    # print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
    plt.figure('Accuracy')
    plt.ylim([0,1])
    plt.plot(accuracy_sum,label='accuracy')
    plt.title('Pytorch_CNN_Accuracy')
    plt.legend()
    plt.show()





def test_img():
    model = torch.load('model')
    for i, (test_x, test_y) in enumerate(test_loader):  #i:batch
        img_x, img_y = test_x, test_y
        break

    img, label = img_x, img_y
    # plt.imshow(img)
    print(img[0][0].shape)
    plot_img = img[0][0]
    plot_img = plot_img.detach().numpy()
    plt.figure('original image')
    plt.imshow(plot_img)

    img = Variable(img)
    label = Variable(label)
    
    pred = model(img)
    print(pred)
    pred = torch.max(pred,1)[1].numpy()
    print('label: ', label.detach().numpy(), ', prediction: ', pred)

    plt.show()








# train()
# test()
test_img()