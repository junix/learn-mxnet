from mx import autograd
from mx import ndarray as nd

a = nd.array((1,))
x = nd.array((-40,))
x.attach_grad()


def SGD():
    x[:] = x - 0.001 * x.grad


def net(a, b):
    return nd.dot(a, nd.abs(b)) - 7


# min ||x| + 7|
def min_x7():
    with autograd.record():
        b = net(a, x)
        diff = b ** 2
    diff.backward()
    SGD()


if __name__ == '__main__':
    for i in range(10000):
        min_x7()
    print(net(a,x))
