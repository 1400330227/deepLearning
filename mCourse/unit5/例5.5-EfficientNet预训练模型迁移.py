import os
import random
import time

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNet.from_pretrained('efficientnet-b7').to(device)

for param in model.parameters():
    param.requires_grad = False


# print(model.eval())


class FlowerDataset(Dataset):
    def __init__(self, data_file_label):
        super().__init__()
        self.data_file_label = data_file_label

    def __len__(self):  # 需要重写该方法，返回数据集大小
        t = len(self.data_file_label)
        return t

    def __getitem__(self, idx):
        fn, label = self.data_file_label[idx][0], self.data_file_label[idx][1]
        img = Image.open(fn).convert('RGB')  # (600, 800, 3)
        img = transform(img)
        return img, label


def getFileLabel(tmp_path):
    dirs = list(os.walk(tmp_path))[0][1]
    L = []
    for label, dir in enumerate(dirs):
        path2 = os.path.join(tmp_path, dir)
        files = list(os.walk(path2))[0: 100]
        for file in files[0][2]:  # files[0][2]为path2目录下的所有文件
            fn = os.path.join(path2, file)
            if os.path.exists(fn):
                t = (fn, label)
                L.append(t)
    return L


path = r'../datasets/data/flower_photos'
file_labels = getFileLabel(path)
random.shuffle(file_labels)
random.shuffle(file_labels)  # 打乱顺序

rate = 0.7
train_length = int(rate * len(file_labels))
train_file_labels = file_labels[:train_length]
test_file_labels = file_labels[train_length:]
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为(224,224)
    transforms.ToTensor(),  # 转化张量
])

batch_size = 128
train_dataset = FlowerDataset(train_file_labels)
train_loader = DataLoader(dataset=train_dataset,  # 打包
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = FlowerDataset(test_file_labels)
test_loader = DataLoader(dataset=test_dataset,  # 打包
                         batch_size=batch_size,
                         shuffle=True)


class FlowerDataSet(Dataset):  # 构建数据集类
    def __init__(self, data_file_label):  #
        self.data_file_label = data_file_label

    def __len__(self):  # 需要重写该方法，返回数据集大小
        t = len(self.data_file_label)
        return t

    def __getitem__(self, idx):
        fn, label = self.data_file_label[idx][0], self.data_file_label[idx][1]
        img = Image.open(fn).convert('RGB')  # (600, 800, 3)
        img = transform(img)
        return img, label


class Net(nn.Module):

    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(2048, 5)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


net = Net(model).to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)




start = time.time()
net.train()

for epoch in range(10):
    ep_loss = 0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = net(x)
        loss = nn.CrossEntropyLoss()(pre_y, y)
        ep_loss += loss.item() * x.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('第 %d 轮循环中，损失函数的平均值为: %.4f' % (epoch + 1, (ep_loss / len(train_loader.dataset))))

end = time.time()

# for epoch in range(100):

def getAccOnadataset(data_loader):
    correct = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            pre_y = net(x)
            pre_y = torch.argmax(pre_y, dim=1)
            t = (pre_y == y).long().sum()
            correct += t
    correct = 1.0 * correct / len(data_loader.dataset)
    net.train()

    return correct.item()

if __name__ == '__main__':
    torch.save(net, 'efficient_model.pt')
    acc_test = getAccOnadataset(test_loader)
