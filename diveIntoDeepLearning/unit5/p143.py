import torch

x, w_xh = torch.randn(3, 1), torch.randn(1, 4)

print(x)
print(w_xh)

h, w_hh = torch.randn(3, 4), torch.randn(4, 4),

a = torch.matmul(x, w_xh) + torch.matmul(h, w_hh)

print(a)

