import torch
# P37----------------------------------------------------
def f(w):
    t = 2*(w-2) ** 2 + 1
    return t
def df(w):
    t= 4*(w-2)
    return t
lr = 0.1
w = torch.tensor([5])
for epoch in range(200):
    w= w-lr*df(w)
y = f(w)
w, y = round(w.item()), round(y.item(), 2)
print(w, y)
