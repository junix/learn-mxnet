from mxnet import ndarray as nd
from mxnet import autograd
from math import fabs

x = nd.array((4,))
x.attach_grad()


def SGD():
    x[:] = x - 0.001 * x.grad


def test():
    a = nd.array((1,))
    with autograd.record():
        b = nd.dot(a, x) + 7
        diff = b ** 2
    diff.backward()
    SGD()


if __name__ == '__main__':
    for i in range(10000):
        test()
    print(x)
