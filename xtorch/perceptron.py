#!/usr/bin/env python

import torch as t
import random
from torch.autograd import Variable
import torch.optim as optim

a = Variable(t.rand(2), requires_grad=True)
b = Variable(t.rand(1), requires_grad=True)

Data = [
    ((-1.0, -2.1), -1.0),
    ((.0, 0.9), 1.0),
    ((1., 0.1), 1.0),
    ((2., 0.9), -1.)
]


class Model:
    def __init__(self):
        self.a = Variable(t.rand(2), requires_grad=True)
        self.b = Variable(t.rand(1), requires_grad=True)

    def parameters(self):
        return (self.a, self.b)

    def input(self, x):
        return self.a.data.dot(x) + self.b.data

    def show_params(self):
        print('new:w0,w1,b={}'.format(t.FloatTensor([self.a.data[0], self.a.data[1], self.b.data[0]])))


def filter_error(model):
    for x, y in Data:
        r = y * model.input(t.Tensor(x))
        if r[0] < 0.0:
            yield x, y


def select_one(xs):
    xs = list(xs)
    random.shuffle(xs)
    return xs[0] if xs else None


def loss(a, b, x, y):
    x = Variable(t.FloatTensor(x))
    return y * (a.dot(x) + b)


def train(model):
    model.show_params()
    optimizer = optim.SGD(params=model.parameters(), lr=-0.01)
    for epoch in range(10000):
        e = select_one(filter_error(model))
        if e is None:
            break
        x, y = e
        optimizer.zero_grad()
        l = loss(model.a, model.b, x, y)
        l.backward()
        optimizer.step()
    model.show_params()


def show_test(model):
    for x, y in Data:
        r = loss(model.a, model.b, t.Tensor(x), y)
        print(r.data[0])


if __name__ == '__main__':
    model = Model()
    train(model)
    show_test(model)
