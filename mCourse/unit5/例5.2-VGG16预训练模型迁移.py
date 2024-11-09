import torch
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(weights=True).to(device)

for param in vgg16.parameters():
    param.requires_grad = False

conv1 = nn.Conv2d(1, 3, 3, padding=0, bias=True, stride=1)
conv2 = vgg16.features[0]
conv3 = vgg16.features[2]
conv4 = nn.Conv2d(64, 512, 3)
conv5 = vgg16.features[28]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv3,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv4,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512*w*h
        )

        self.avgPool2d = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 6 * 6, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgPool2d(x)  # 512*6*6
        x = x.reshape(x.size(0), -1)  # batch*18,432
        x = self.classifier(x)  # batch*2
        return x


net = Net().to(device)
x = torch.randn(16, 1, 224, 224).to(device)  # 随机产生测试数据

print(vgg16.eval())
print(net.eval())
# conv1 = vgg16.features
