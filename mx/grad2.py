from mx import autograd
from mx import ndarray as nd

x = nd.array((3, 4))
x.attach_grad()


def SGD():
    x[:] = x - 0.001 * x.grad


def f(x):
    """
    f([x1,x2]) = |x1|**2 + |x2|**3
    ▽f = [2*x1, 3*x2**2]
    """
    x0 = nd.abs(x)
    return x0[0] ** 2 + x0[1] ** 3 + 5


def cal_grad():
    with autograd.record():
        y = f(x)
    y.backward()
    # print(x.grad)
    SGD()


if __name__ == '__main__':
    for i in range(10000):
        cal_grad()
    print(f(x))  # 5
