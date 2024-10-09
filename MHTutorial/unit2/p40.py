import torch
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 		#读取数据
Y = [-9.51, -5.74, -2.84, -1.8, 0.54, 1.51, 4.33, 7.06, 9.34, 10.72]

x = torch.tensor(X)
y = torch.tensor(Y)

w,b = torch.rand(1), torch.rand(1)

def f(x):
    t= w* x + b
    return t

def dw(x, y):
    t= (f(x) -y ) * x
    return t

def db(x, y):
    t = (f(x) - y)
    return t

lr = torch.tensor([0.01])
for epoch in range(2000):
    for x,y in zip(X,Y):
        dw_v, db_v = dw(x, y), db(x, y)
        w = w - lr*dw_v
        b = b - lr*db_v
print(w,b)

plt.scatter(X,Y,c='r')
X2 = [X[0],X[len(X)-1]]   		#过两点绘制感知器函数直线图
Y2 = [f(X[0]),f(X[len(X)-1])]
plt.plot(X2,Y2,'--',c='b')
plt.tick_params(labelsize=13)
plt.show()