#!/usr/bin/env python

import torch as t
import random
from torch.autograd import Variable
import torch.optim as optim

a = Variable(t.rand(2), requires_grad=True)
b = Variable(t.rand(1), requires_grad=True)
sgd = optim.SGD(params=[a, b], lr=-0.01)

Data = [
    ((-1.0, -2.1), -1.0),
    ((.0, 0.9), 1.0),
    ((1., 0.1), 1.0),
    ((2., 0.9), -1.)
]


def filter_error():
    for x, y in Data:
        r = loss(x, y)
        if r.data[0] < 0.0:
            yield x, y


def select_one(xs):
    xs = list(xs)
    random.shuffle(xs)
    return xs[0] if xs else None


def loss(x, y):
    x = Variable(t.FloatTensor(x))
    return y * (a.dot(x) + b)


def loss_batch(xs):
    return sum([loss(x, y) for x, y in xs])


def train():
    show_params()
    for epoch in range(10000):
        e = select_one(filter_error())
        if e is None:
            break
        x, y = e
        sgd.zero_grad()
        l = loss(x, y)
        l.backward()
        sgd.step()
    show_params()


def show_test():
    for x, y in Data:
        r = loss(x, y)
        print(r.data[0])


def show_params():
    print('new:w0,w1,b={}'.format(t.FloatTensor([a.data[0], a.data[1], b.data[0]])))


if __name__ == '__main__':
    train()
    show_test()
