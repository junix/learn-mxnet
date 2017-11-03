import torch
from torch.autograd import Variable

A = Variable(torch.FloatTensor([1, 2]).view(1, 2))
x = Variable(torch.FloatTensor([1, 2]).view(2, 1),requires_grad=True)


# ((Ax).t * (Ax)) ** 2
ax = A.mm(x)
y = ax.t().mm(ax)
y1 = y.pow(2)

y1.backward()

print(x.grad.data)
