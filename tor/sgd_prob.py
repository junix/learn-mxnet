import torch
from torch.autograd import Variable

p = Variable(torch.DoubleTensor([0.59]), requires_grad=True)
k = 3
n = 5

# 伯努利概率P(5,3)
def f():
    return -3 * p.log() - 2 * (1 - p).log()


# for i in range(10000):
last_p = None
while True:
    y = f()
    y.backward()
    if last_p is not None and y.data[0] >= last_p:
        break
    last_p = y.data[0]
    p.data.sub_(1e-6 * p.grad.data[0])
    p.grad.data.zero_()
    p.data.clamp(min=0.0, max=1.0)

print(p.data[0])
