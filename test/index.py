import numpy as np
import torch

x = np.random.normal(0, 0.1, size=(3, 4))
x = torch.tensor(x, device='cpu', dtype=torch.float)
y = torch.nn.Parameter(x, requires_grad=True)
print(y)

# print

# print(x.shape)
