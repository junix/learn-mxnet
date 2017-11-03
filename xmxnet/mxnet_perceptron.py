#!/usr/bin/env python

from xmxnet import ndarray as nd
import xmxnet.autograd as ag

D = [
    (-1, -2.1, -1),
    (0, 0.9, 1),
    (1, 0.1, 1),
    (2, 0.9, -1)
]


def net(a, b, x, y, c):
    return nd.sum(- c * (y - (a * x - b)))


if __name__ == '__main__':
    a = nd.array([-1])
    b = nd.array([-9])
    a.attach_grad()
    b.attach_grad()
    x = nd.array(D)[:, 0]
    y = nd.array(D)[:, 1]
    c = nd.array(D)[:, 2]
    with ag.record():
        l = net(a, b, x, y, c)
        l.backward()
        print(b.grad)
        print(l)
