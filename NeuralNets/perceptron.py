import gzip, idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim



# 读取mnist数据集
images_archive = gzip.open('train-images-idx3-ubyte.gz', 'r')
labels_archive = gzip.open('train-labels-idx1-ubyte.gz', 'r')
test_img_archive = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
test_lbl_archive = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')

images = idx2numpy.convert_from_file(images_archive)
labels = idx2numpy.convert_from_file(labels_archive)
test_images = idx2numpy.convert_from_file(test_img_archive)
test_labels = idx2numpy.convert_from_file(test_lbl_archive)

# print(images.shape, labels.shape, test_images.shape, test_labels.shape)

# 可视化数据集
def plot_img(image, label):
  plt.imshow(image)
  plt.title(label)
  plt.show()

# plot_img(images[0], labels[0])

# 拆分数据集
#splitting the data 80-20
X_train = images[:48000]
Y_train = labels[:48000]
X_val = images[48000:]
Y_val = labels[48000:]

# print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

class ImagesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])


train = ImagesDataset(X_train, Y_train)
val = ImagesDataset(X_val, Y_val)
test = ImagesDataset(X_val, Y_val)
# test = ImagesDataset(test_images, test_labels)

class Perceptron(nn.Module):
  def __init__(self):
    super(Perceptron, self).__init__()
    self.l1 = nn.Linear(784, 128)
    self.l2 = nn.Linear(128, 10)

  def forward(self, x):
    x = x.view(-1, 784)
    x = F.relu(self.l1(x))
    x = self.l2(x)
    return F.log_softmax(x, dim=-1)

train_loader = DataLoader(train, batch_size=256, shuffle=True)
eval_loader = DataLoader(val, batch_size=256, shuffle=True)
test_loader = DataLoader(test, batch_size=256)

device = 'cpu'
model = Perceptron()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
cel = nn.CrossEntropyLoss()


def fit(epochs, lr):
    optimizer.param_groups[0]['lr'] = lr
    train_loss = []
    eval_loss = []
    for epoch in range(1, epochs + 1):
        val_loss = 0

        # train loop
        model.train()
        for batch_index, (x, y) in enumerate(train_loader):
            x, y = x.float().to(device), y.long().to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = cel(pred, y)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        # val loop
        model.eval()
        for batch_index, (x, y) in enumerate(eval_loader):
            x, y = x.float().to(device), y.long().to(device)
            pred = model(x)
            val_loss += cel(pred, y)

        val_loss /= len(eval_loader.dataset)
        eval_loss.append(val_loss.item())

        print(f'epoch: {epoch};  loss:{round(loss.item(), 4)}\t val_loss: {val_loss}')
        #save model
    torch.save(model, 'model.pkl')
    return train_loss, eval_loss




#保存模型
#测试-加载模型
def test(model_path):
    model = torch.load(model_path)
    for batch_index, (x, y) in enumerate(test_loader):
        x, y = x.float().to(device), y.long().to(device)
        optimizer.zero_grad()
        pred = model(x)
        pred = torch.argmax(pred, -1)
        # print(pred)
        # print(y)
        correct = torch.sum(pred == y)
        pct_correct = correct / torch.tensor(test_loader.batch_size, dtype=torch.float32)
        print(pct_correct)


train_loss, eval_loss = fit(200, 1e-4)
x = np.arange(0, 200, 1)
# print(train_loss, eval_loss)
plt.plot(x, eval_loss)
plt.show()

modl_path = 'model.pkl'
# test(modl_path)