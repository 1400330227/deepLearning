import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

# x = np.random.normal(0, 0.1, size=(3, 4))
# x = torch.tensor(x, device='cpu', dtype=torch.float)
# y = torch.nn.Parameter(x, requires_grad=True)
# print(y)

# print

# print(x.shape)


# x, y = torch.randn(2040, 4), torch.randn(2040),
# print(x.shape, y.shape)
#
# train_x, train_y = x[:500], y[:500]
#
# print(train_x.shape, train_y.shape)


# train_x, train_y = torch.randn(140, 3), torch.randn(140),
# train_set = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
#
# print(len(train_set))
# print(len(train_loader))
# for i, (x, y) in enumerate(train_loader):
#     print(i, x.shape, y.shape)


import torch
from torch import nn

rnn = nn.GRU(10, 20, 2, bidirectional=True)
input = torch.randn(5, 3, 10)
output, hn = rnn(input)

print(output.shape, hn.shape)
