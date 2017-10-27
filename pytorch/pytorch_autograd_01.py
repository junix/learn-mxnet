import pytorch
import pytorch.optim as optim
from pytorch.autograd import Variable

a = Variable(pytorch.FloatTensor([1.]))
b = Variable(pytorch.FloatTensor([7.]))


def net(x):
    y = x * x + b
    return y


if __name__ == '__main__':
    x0 = Variable(pytorch.FloatTensor([-1.]), requires_grad=True)
    y = 0

    optimizer = optim.SGD([x0], lr=0.01)
    optimizer.zero_grad()
    for i in range(10000):
        y = net(x0)
        loss = y
        loss.backward()
        optimizer.step()
    print(y)
