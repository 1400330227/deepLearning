import torch

# 教材P6
# --------------------------------------------------------
print("hello world")
print(torch.cuda.is_available())
x0 = torch.tensor([2])
x = torch.tensor([x0])
print(x0)
# --------------------------------------------------------


# 教材p8
# --------------------------------------------------------
x0 = torch.tensor(2)  # 0阶张量，形状为torch.Size([])，亦写为()
x1 = torch.tensor([2])  # 1阶张量，形状为torch.Size([1])，亦写为(1)
x2 = torch.tensor([2, 3])  # 1阶张量，形状为torch.Size([2])，亦写为(2)
x3 = torch.tensor([[2, 3, 4],  # 2阶张量，形状为torch.Size([2, 3])，亦写为(2, 3)
                   [5, 6, 7]])
x4 = torch.tensor([[2, 3, 4],  # 2阶张量，形状为torch.Size([3, 3])，亦写为(3, 3)
                   [5, 6, 7],
                   [8, 9, 10]])

print('x0 的阶数为： {}，形状为：{}'.format(x0.ndim, x0.size()))
print('x1 的阶数为： {}，形状为：{}'.format(x1.ndim, x1.size()))
print('x2 的阶数为： {}，形状为：{}'.format(x2.ndim, x2.size()))
print('x3 的阶数为： {}，形状为：{}'.format(x3.ndim, x3.size()))
print('x4 的阶数为： {}，形状为：{}'.format(x4.ndim, x4.size()))
# --------------------------------------------------------


# 教材p9
# --------------------------------------------------------
x5 = torch.randn(32, 3, 224, 22)  # 标准正态分布
x6 = torch.rand(32, 3, 224, 22)  # 0-1均匀分布
x7 = torch.randint(0, 6, [32, 3, 224, 22])  # 0-6随机抽取
print('x5 的阶数为： {}，形状为：{}'.format(x5.ndim, x5.size()))
print('x6 的阶数为： {}，形状为：{}'.format(x6.ndim, x6.size()))
print('x7 的阶数为： {}，形状为：{}'.format(x7.ndim, x7.size()))
# --------------------------------------------------------

# 教材p10
# --------------------------------------------------------
x8 = torch.randint(0, 10, [4, 10])

print(x8)
print(x8[:, 3:8:2])
print(x8[:, 1::1])
# print(x8[:, :, 1::1])

# --------------------------------------------------------


# 教材p10
# --------------------------------------------------------
x9 = torch.randint(0, 6, [2, 3, 4])
print(x9)
print(x9.sum())  # 求x中所有元素之和
print(x9.sum(dim=0))  # 沿着第1维进行相加
print(x9.sum(dim=1))  # 沿着第2维进行相加
# --------------------------------------------------------

x10 = torch.randint(-6, 6, [2, 3])
print(x10)
print(x10.min())
print(x10.min(dim=0))
print(x10.min(dim=0)[0])
print(x10.min(dim=1))

x11 = torch.randint(-6, 6, [2, 3])
print(x11)
print(x11.float().mean())
print(x11.float().mean(dim=0))
print(x11.float().mean(dim=1))
#

x12 = torch.randint(0, 6, [10, 20])
y = x12.reshape(10, 4, 5)  # 等价于y = x.view(10,4,5)
print(x12.size())
print(y.size())

x13 = x12.reshape(1, 1, - 1)
print(x13.size())

x = torch.randint(0, 6, [10, 20])
y1 = x.unsqueeze(0)  # 增加第1维，维的长度为1
y2 = x.unsqueeze(1)  # 增加第2维，维的长度为1
print(x.shape)
print(y1.shape)
print(y2.shape)
print('-----------------')
x = torch.randint(0, 6, [1, 1, 1, 10, 20])
y3 = x.squeeze(2)  # 去掉第3维
y4 = x.squeeze(3)  # 无效，因为第4维的长度不是1
y5 = x.squeeze()  # 去掉x中所有长度为1的维
print(x.shape)
print(y3.shape)
print(y4.shape)
print(y5.shape)

print('-----------------')
x = torch.randint(0, 6, [2, 4])
y = x.t()  # 交换第1维和第2维 （只适用于2阶张量）
print(x.shape, y.shape)
x = torch.randint(0, 6, [2, 4, 6, 8])
y = x.transpose(0, 2)  # 交换第1维和第3维
print(x.shape, y.shape)
y = x.permute(0, 2, 1, 3)
print(y.shape)
print(y.size())

print('-----------------')
x = torch.randint(0, 6, [4])
y = torch.randint(-5, 6, [4])
z = torch.dot(x, y)
print(x)
print(x[0])
print(y)
print(z)

print('-----------------')
x = torch.randint(0, 5, [5, 7, 2, 3])  # 35个2×3矩阵（先分为5组，再分为7组）
y = torch.randint(-2, 3, [5, 7, 3, 4])  # 35个3×4矩阵（先分为5组，再分为7组）
z = torch.matmul(x, y)
print(x.shape, '*', y.shape, '--->', z.shape)

A = torch.rand(2, 3, 4)  # 张量 A 的形状为 (2, 3, 4)
B = torch.rand(4)  # 张量 B 的形状为 (4,)
result = A * B  # 广播机制会将 B 扩展为 (1, 1, 4)，然后再扩展为 (2, 3, 4)
print("A.shape:", A.shape)  # A.shape: (2, 3, 4)
print("B.shape:", B.shape)  # B.shape: (4,)
print("result.shape:", result.shape)  # 逐元素乘法操作成功进行，结果的形状为 (2, 3, 4)

x = torch.tensor([3.], requires_grad=True)
y = torch.tensor([2.], requires_grad=True)

z = 2 * x ** 2 - 6 * y ** 2
f = z ** 2

f.backward()

print('f的值为: {}'.format(f.item()))
print('f关于x梯度的值为: {}'.format(x.grad.item()))
print('f关于y梯度的值为: {}'.format(y.grad.item()))


x1 = torch.randint(0,6,[3,4])
x2 = torch.randint(0,6,[3,2])
x = torch.cat([x1,x2],dim=1)
print(x1)
print(x2)
print(x)

