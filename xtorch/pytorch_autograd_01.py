import xtorch
import xtorch.optim as optim
from xtorch.autograd import Variable

a = Variable(xtorch.FloatTensor([1.]))
b = Variable(xtorch.FloatTensor([7.]))


def net(x):
    y = x * x + b
    return y


if __name__ == '__main__':
    x0 = Variable(xtorch.FloatTensor([-1.]), requires_grad=True)
    y = 0

    optimizer = optim.SGD([x0], lr=0.01)
    optimizer.zero_grad()
    for i in range(10000):
        y = net(x0)
        loss = y
        loss.backward()
        optimizer.step()
    print(y)
