import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

# -----------------------------------------------------
seq_len = 4  # 序列长度(每个序列有4个元素，1个元素是一年，12个月，即12个数据表示一个向量，构成一个元素)
vec_dim = 12  # 序列中每个元素的特征数目。本程序采用的序列元素为一年的旅客，一年12个月，即12维特征。

data = read_csv(r'../data/data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=0)
data = np.array(data)  # (144, 1)
data2 = data[:, 0]
sc = MinMaxScaler()
data = sc.fit_transform(data)  # 归一化
data = data.reshape(-1, vec_dim)  # torch.Size([12, 12])
train_x, train_y = [], []

train_x = torch.FloatTensor(train_x)  # torch.Size([8, 4, 12]) torch.Size([8, 12])
train_y = torch.FloatTensor(train_y)


# ------------------------------------------------------
class Air_Model(nn.Module):
    def __init__(self, n=vec_dim, s=128, m=vec_dim):
        super(Air_Model, self).__init__()
        self.s = s
        self.U = nn.Linear(n, s)
        self.V = nn.Linear(s, m)
        self.W = nn.Linear(s, s)

    def forward(self, x):  # torch.Size([1, 4, 12])
        a_t_1 = torch.rand(x.size(0), self.s)
        lp = x.size(1)
        for k in range(lp):
            input1 = x[:, k, :]
            input1 = self.U(input1)
            input2 = self.W(a_t_1)
            input = input1 + input2

            a_t = nn.ReLU(input)
            a_t_1 = a_t

        y_t = self.V(a_t)
        return y_t


air_Model = Air_Model()
optimizer = torch.optim.Adam(air_Model.parameters(), lr=0.01)

for ep in range(400):
    for i, (x, y) in enumerate(zip(train_x, train_y)):
        x = x.unsqueeze(0)  # 加上批   torch.Size([1, 4, 12])
        pre_y = air_Model(x)  # torch.Size([1, 12])
        pre_y = torch.squeeze(pre_y)  # torch.Size([12])
        loss = torch.nn.MSELoss()(pre_y, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            print('epoch:{:3d}, loss:{:6.4f}'.format(ep, loss.item()))

# ---------------------------------------

torch.save(air_Model, 'air_Model')

'''
'''
air_Model = torch.load('air_Model')
air_Model.eval()
pre_data = []
for i, (x, y) in enumerate(zip(train_x, train_y)):
    x = x.unsqueeze(0)  # 加上批   torch.Size([1, 4, 12])
    pre_y = air_Model(x)  # torch.Size([1, 12])
    pre_data.append(pre_y.data.numpy())
    # print(pre_y.data.numpy())

# ------------------------


plt.figure()
pre_data = np.array(pre_data)  # (8, 1, 12)
pre_data = pre_data.reshape(-1, 1).squeeze()  # (8, 12) ---> (96,)

x_tick = np.arange(len(pre_data)) + (seq_len * vec_dim)
plt.plot(list(x_tick), pre_data, linewidth=2.5, label='预测数据')  # 从48开始
# ------
ori_data = data.reshape(-1, 1).squeeze()  # (144,)

plt.plot(range(len(ori_data)), ori_data, linewidth=2.5, label='原始数据')  # 据'

# plt.rcParams['font.sans-serif']=['SimHei']
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.ylabel("数据的大小（已归一化）", fontsize=14)  # Y轴标签

plt.xlabel("月份的序号", fontsize=14)  # Y轴标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签simhei
plt.grid()
plt.show()

exit(0)

# 绘制原始数据的曲线图=============================
plt.figure()

# ------
ori_data = data.reshape(-1, 1).squeeze()  # (144,)

plt.plot(range(len(data2)), data2, linewidth=2.5)  # 据'

# plt.rcParams['font.sans-serif']=['SimHei']

plt.tick_params(labelsize=14)
plt.ylabel("数据的大小", fontsize=14)  # Y轴标签

plt.xlabel("月份的序号", fontsize=14)  # Y轴标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签simhei
plt.grid()
plt.show()
