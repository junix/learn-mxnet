from torch.autograd import Variable
import torch

# x^3 + y^2
a = Variable(torch.DoubleTensor((1.0, 1.0)), requires_grad=True)
X = Variable(torch.DoubleTensor([2, 3]), requires_grad=True)
Y = Variable(torch.DoubleTensor([1, 2]), requires_grad=True)


def net(x):
    return a[0] * (x[0] ** 3) + a[1] * (x[1] ** 2)


y0 = net(X)
y1 = net(Y)
# y0.backward()
# a.grad.data.zero_()
o = torch.FloatTensor(2)
y1.backward(gradient=o)

# print(a.grad)
print(Y.grad)
