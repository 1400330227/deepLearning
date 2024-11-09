import os
import sys
import time

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为(224,224)
    transforms.ToTensor()
])


class CatDog_Dataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = os.listdir(dir)[0: 100]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        fn = os.path.join(self.dir, file)
        img = Image.open(fn).convert('RGB')
        img = transform(img)

        y = 0 if 'cat' in file else 1

        return img, y


batch_size = 32
train_dir = '../datasets/data/catdog/training_set'
test_dir = '../datasets/data/catdog/test_set'

train_dataset = CatDog_Dataset(train_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CatDog_Dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
print('训练集大小：', len(train_loader.dataset))
print('测试集大小：', len(test_loader.dataset))


class Model_CatDog(nn.Module):

    def __init__(self):
        super(Model_CatDog, self).__init__()

        # Conv2d -> input = batch, input_channels, height, width; output = batch, out_channels,height, width
        self.features = nn.Sequential(
            # 224*224*3 -> (224-5+2*2)/1+1 = 224, 224*224*64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),  # 224*224*64
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224*224*3 -> (224-2+2*0)/2+1 = 112, 112*112*64

            # 112*112*64 -> (112-5+2*0)/1+1 = 108, 108*108*128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 108*108*128 -> (108-2+2*0)/2+1 = 54, 54*54*128

            # 54*54*128 -> (54-3+2*0)/1+1 = 52, 52*52*128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 52*52*128 -> (52-2+2*0)/2+1 = 52, 26*26*128

            # 26*26*128 -> (26-3+2*0)/1+1 = 24, 24*24*256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24*24*256 -> (24-2+2*0)/2+1 = 52, 12*12*256
        )

        self.avgPool2d = nn.AdaptiveAvgPool2d((12, 12))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=12 * 12 * 256, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=2, bias=True),
        )

    def forward(self, x):
        x = self.features(x)  # batch*12*12*256
        x = self.avgPool2d(x)
        x = x.reshape(x.size(0), -1)  # batch*36864
        x = self.classifier(x) # batch*2
        return x


model_CatDog = Model_CatDog().to(device)
optimizer = torch.optim.SGD(model_CatDog.parameters(), lr=0.001, momentum=0.9)
start = time.time()  # 开始计时
model_CatDog.train()

for epoch in range(2):
    ep_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = model_CatDog(x)
        loss = nn.CrossEntropyLoss()(pre_y, y.long())  # 使用交叉熵损失函数
        ep_loss += loss * x.size(0)  # loss是损失函数的平均值，故要乘以样本数量
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('第 %d 轮循环中，损失函数的平均值为: %.4f' % (epoch + 1, (ep_loss / len(train_loader.dataset))))

end = time.time()  # 计时结束
print('训练时间为:  %.1f 秒 ' % (end - start))
torch.save(model_CatDog, 'model_CatDog.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model_CatDog.pt')
print(model.eval())

with torch.no_grad():
    correct = 0
    for i, (x, y) in enumerate(train_loader):
        x, y, = x.to(device), y.to(device)
        pre_y = model(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t

t = 1. * correct / len(train_loader.dataset)

print('1、网络模型在训练集上的准确率：{:.2f}%'.format(100 * t.item()))

correct = 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        pre_y = model(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t
t = 1. * correct / len(test_loader.dataset)
print('2、网络模型在测试集上的准确率：{:.2f}%'.format(100 * t.item()))
