#!/usr/bin/env python

import torch as t
import random
from torch.autograd import Variable
import torch.optim as optim

a = Variable(t.rand(2), requires_grad=True)
b = Variable(t.rand(1), requires_grad=True)

D = [
    ((-1.0, -2.1), -1.0),
    ((.0, 0.9), 1.0),
    ((1., 0.1), 1.0),
    ((2., 0.9), -1.)
]


def filter_error():
    a0, b0 = a.data, b.data
    for x, y in D:
        x = t.Tensor(x)
        r = y * (a0.dot(x) + b0)
        if r[0] < 0.0:
            yield x, y


def net(x, y, c):
    return - c * (a * x + b)


def net1(xs):
    return sum([net(x, y, c) for x, y, c in xs])


def select_one(xs):
    random.shuffle(xs)
    return xs[0] if xs else None


if __name__ == '__main__':
    opt = optim.SGD(params=[a, b], lr=0.1)

    print('orig:{} : {}'.format(a.data[0], b.data[0]))
    for epoch in range(100000):
        opt.zero_grad()
        es = list(filter_error())
        random.shuffle(es)

        if es:
            # l = net1(es)
            # l.backward()
            # print("a={},b={},{}, a_grad={},b_grad={}".format(
            #     a.data[0], b.data[0], es, a.grad.data[0], b.grad.data[0]))
            x, y = es[0]
            a.data.add_(0.01 * x * y)
            b.data.add_(0.01 * y)
        else:
            print('new:{} : {}'.format(a.data, b.data))
            a0, b0 = a.data, b.data
            for x, y in D:
                x = t.Tensor(x)
                r = y*(a0.dot(x) + b0)
                print(r[0])
            # print(list(filter_error()))
            break
