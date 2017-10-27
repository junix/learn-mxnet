import torch
from torch.autograd import Variable
import torch.optim as optim

# <<deep learning>> 4.5 example


class Model:
    a = torch.FloatTensor((1.0, 2.0, 3.0))
    b = torch.FloatTensor((60,))

    def __init__(self):
        self.x = Variable(torch.rand(3), requires_grad=True)

    def parameters(self):
        return (self.x,)

    def input(self):
        return self.x

    def show(self):
        print('x={}'.format(self.x.data.numpy()))
        print('loss={}'.format(loss(self.x).data.numpy()[0]))


def loss(x):
    va, vb = Variable(Model.a), Variable(Model.b)
    y = va.dot(x) - vb
    return y.dot(y) * 0.5


def sgd_small_enough(y):
    return y.data[0] < 1


def train(model):
    sgd = optim.SGD(model.parameters(), lr=0.001)
    for _ in range(200000):
        sgd.zero_grad()
        y = loss(model.input())
        y.backward()
        sgd.step()
        if sgd_small_enough(y):
            break


if __name__ == '__main__':
    model = Model()
    train(model)
    model.show()
