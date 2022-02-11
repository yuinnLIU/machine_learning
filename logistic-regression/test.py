import numpy as np
import torch

train_y = torch.Tensor([0,0,1,1])
y_pred = [0.2,0.4,0.9,0.6]
y_pred = torch.Tensor(y_pred)
res = y_pred.ge(0.5).float().squeeze()


correct = (res == train_y).sum()  # 统计正确预测的样本个数
acc = correct.item() / train_y.size(0)  # 计算分类准确率

print(acc)