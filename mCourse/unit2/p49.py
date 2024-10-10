import torch
import torch.nn as nn

from MHTutorial.unit2.p40 import X

# 读入数据
X1 = [2.49, 0.50, 2.73, 3.47, 1.38, 1.03, 0.59, 2.25, 0.15, 2.73]
X2 = [2.86, 0.21, 2.91, 2.34, 0.37, 0.27, 1.73, 3.75, 1.45, 3.42]
Y = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]  # 类标记

X1 = torch.tensor(X1)
X2 = torch.tensor(X2)
X = torch.stack((X1,X2),dim=1)
Y = torch.tensor(Y)

class Perceptron(nn.Module):

    def __init__(self):
        super(Perceptron, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.0]))
        self.w2 = nn.Parameter(torch.tensor([0.0]))
        self.b = nn.Parameter(torch.tensor([0.0]))

    def f(self, x):
        x1, x2 = x[0], x[1]
        t = self.w1 * x1 + self.w2 * x2 + self.b
        z = 1.0 / (1 + torch.exp(t))
        return z

    def forward(self, x):
        pre_y = self.f(x)
        return pre_y


perceptron = Perceptron()

optimizer = torch.optim.Adam(perceptron.parameters(), lr=0.1)

for epoch in range(100):
    for (x, y) in zip(X, Y):
        pre_y = perceptron(x)
        y = torch.Tensor([y])
        loss = nn.BCELoss()(pre_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

s = '学习到的感知器：pre_y = sigmoid(%0.2f*x1 + %0.2f*x2 + %0.2f)' %(perceptron.w1,perceptron.w2,perceptron.b)
print(s)
