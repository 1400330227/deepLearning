import os
import time

import torch
from PIL import Image
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CatDog_Dataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.file = os.listdir(self.dir)[0: 1000]

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        file = self.file[idx]
        fn = os.path.join(self.dir, file)
        img = Image.open(fn).convert('RGB')
        img = transforms(img)
        y = 0 if 'cat' in file else 1
        return img, y


batch_size = 100
train_dir = '../datasets/data/catdog/training_set'
test_dir = '../datasets/data/catdog/test_set'

train_dataset = CatDog_Dataset(train_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CatDog_Dataset(test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

cat_dog_vgg16 = models.vgg16(pretrained=True)

for i, param in enumerate(cat_dog_vgg16.parameters()):
    param.requires_grad = False

cat_dog_vgg16.classifier[3] = nn.Linear(4096, 1024)
cat_dog_vgg16.classifier[6] = nn.Linear(1024, 2)

cat_dog_vgg16.train()
cat_dog_vgg16 = cat_dog_vgg16.to(device)

optimizer = torch.optim.SGD(cat_dog_vgg16.parameters(), lr=0.001, momentum=0.9)

cat_dog_vgg16.train()

start = time.time()
for epoch in range(2):
    ep_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = cat_dog_vgg16(x)
        loss = CrossEntropyLoss()(pre_y, y.long())
        ep_loss += loss * x.shape[0]
        optimizer.zero_grad()  # 对参数的梯度清零
        loss.backward()  # 反向传播并计算各个参数的梯度
        optimizer.step()  # 利用梯度更新参数
    print('第%d轮循环中，损失函数的平均值为：%.4f' % (epoch + 1, ep_loss / len(train_loader.dataset)))

end = time.time()

print('训练事件为：%.1f秒' % (end - start))

torch.save(cat_dog_vgg16, 'cat_dog_vgg16.pt')
model = torch.load('cat_dog_vgg16.pt')
correct = 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        pre_y = model(x)
        pre_y = torch.argmax(pre_y, dim=1)
        t = (pre_y == y).long().sum()
        correct += t

correct = 1. * correct / len(test_loader.dataset)
print('1、网络模型在训练集上的准确率：{:.2f}%'.format(100 * correct.item()))
