import torch

# x = torch.tensor(3)
x = torch.tensor([[2, 1, 4, 3],
                  [1, 2, 3, 4],
                  [4, 3, 2, 1]])
print(x[:,0])
print(x[:,[0]])
print(x[:,[0,1]])
# y = x[:,0]
# x = x[:,[0, 1]]
#
# print(y)
# print(x)
# print(x.shape)
# x = x.view(1, 12)
# x = x.view(12)
# x = torch.tensor((3, 4))

# print(x.shape)
# print(x)

# x = torch.tensor([[0, 1, 2, 3],
#                   [4, 5, 6, 7],
#                   [8, 9, 10, 11]])
# x[1:3, :] = 12
#
# print(x)


# x = torch.tensor((3, 4))

# x1 = torch.randn(32, 3, 224, 224)
# # print('x1的形状: %s' %(x1.shape))
# # print(x1)
# print(x1.shape)
# print(x1.shape[0])
# print(x1.shape[1])

