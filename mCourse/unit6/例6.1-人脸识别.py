import os
import time
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_path = '../datasets/data/faces/training'

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class FaceDataset(Dataset):
    def __init__(self, fn_labels2):
        self.fn_labels = fn_labels2

    def __len__(self):
        return len(self.fn_labels)

    def __getitem__(self, idx):
        img1, label1 = self.fn_labels[idx]
        fg = random.randint(0, 1)  # 随机生成0或1
        if fg == 0:  # 生成同类的三元组
            k = idx + 1
            while True:
                if k >= len(self.fn_labels):
                    k = 0
                img2, label2 = self.fn_labels[k]
                k += 1
                if int(label1) == int(label2):
                    break
        else:  # 生成不同类的三元组
            k = idx + 1
            while True:
                if k >= len(self.fn_labels):
                    k = 0
                img2, label2 = self.fn_labels[k]
                k += 1
                if int(label1) != int(label2):
                    break

        img1 = Image.open(img1)
        img1 = np.array(img1)
        img1 = torch.Tensor(img1)
        img1 = transform(img1)

        img2 = Image.open(img2)
        img2 = np.array(img2)
        img2 = torch.Tensor(img2)
        img2 = transform(img2)

        label = torch.Tensor(np.array([int(label1 != label2)], dtype=np.float32))
        return img1, img2, label


def getFn_Dir(root):
    dir = os.listdir(root)
    file_labels = []
    for i, dir in enumerate(dir):
        label = i
        path2 = os.path.join(root, dir)
        files = os.listdir(path2)

        for file in files:
            fn = os.path.join(path2, file)
            if os.path.exists(fn):
                t = (fn, label)
                file_labels.append(t)

    return file_labels


fn_labels = getFn_Dir(dir_path)
faceDataset = FaceDataset(fn_labels)
train_loader = DataLoader(faceDataset, batch_size=8, shuffle=True)

vgg19 = models.vgg19(weights=True)

for param in vgg19.parameters():
    param.requires_grad = False

# print(vgg19.eval())
vgg19_features = vgg19.features


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0),
            vgg19_features,
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
        )

    def pre_forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # batch * 512
        x = self.classifier(x)
        return x

    def forward(self, img1, img2):
        out1 = self.pre_forward(img1)
        out2 = self.pre_forward(img2)

        return out1, out2


# loss = (1-label)(y1-y2) + label*(max(c - (y1-y2), 0))
# label=0，表示x1和x2表示同一个人的照片，label=1表示不是同一个人的照片

class LossFunction(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, i1, i2, y):
        # 欧氏距离：两个数字相减，平方，sqrt，没有除以n。形状必须是两个维度以上
        dist = torch.pairwise_distance(i1, i2, keepdim=True)
        loss = torch.mean((1 - y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss


siameseNet = SiameseNet().to(device)
optimizer = optim.Adam(siameseNet.parameters(), lr=0.001)
lossFunction = LossFunction()

start = time.time()
for epoch in range(10):
    epoch_loss = 0
    for i, (b_img1, b_img2, b_label) in enumerate(train_loader):
        b_img1, b_img2, b_label, = b_img1.to(device), b_img2.to(device), b_label.to(device)
        pre_o1, pre_o2 = siameseNet(b_img1, b_img2)  # torch.Size([8, 5]) torch.Size([8, 5])
        loss = lossFunction(pre_o1, pre_o2, b_label)
        epoch_loss += loss.item() * b_img1.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

end = time.time()
print('训练耗时：', round((end - start) / 60.0, 1), '分钟')

torch.save(siameseNet, 'siameseNet.pt')  # 保存模型
siameseNet = torch.load('siameseNet.pt')  # 加载训练的模型


def getImg(fn):
    img = Image.open(fn)
    img = np.array(img)
    img = torch.Tensor(img)
    img = transform(img)
    return img


def getImg_show(fn):
    img = Image.open(fn)
    img = img.convert('RGB')  # 用于显示
    img = np.array(img)
    return img


path = '../datasets/data/faces/testing'
fn_labels = getFn_Dir(path)

correct = 0

for fn, label in fn_labels:
    img = getImg(fn).unsqueeze(0).to(device) # 设置batch为1
    img_min, dist_min, label_min, fn_min = -1, 1000, -1, -1
    for fn2, label2 in fn_labels:
        if fn == fn2:
            continue
        img2 = getImg(fn2).unsqueeze(0).to(device)
        pre_o1, pre_o2 = siameseNet(img, img2)
        dist = torch.pairwise_distance(pre_o1, pre_o2, keepdim=True)
        if dist_min > dist.item():
            dist_min = dist.item()
            img_min = img2
            label_min = label2
            fn_min = fn2
    # img,img_min
    correct += int(label == label_min)
    # print(label,label_min)
    img_show = getImg_show(fn)
    img_show2 = getImg_show(fn_min)

    images = dict()
    images[fn] = img_show
    images[fn_min] = img_show2


    def showTwoImages(images, stitle='', rows=1, cols=1):
        figure, ax = plt.subplots(nrows=rows, ncols=cols)
        for idx, title in enumerate(images):
            ax.ravel()[idx].imshow(images[title])
            ax.ravel()[idx].set_title(title)
            ax.ravel()[idx].set_axis_off()
        plt.tight_layout()
        plt.suptitle(stitle, fontsize=18, color='red')
        plt.show()


    stitle = 'Similarity: %.2f' % (dist_min)

    showTwoImages(images, stitle, 1, 2)
print('一共测试了{:.0f}张图片，准确率为{:.1f}%' \
      .format(len(fn_labels), 100. * correct / len(fn_labels)))
