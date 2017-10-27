import tor
from tor.autograd import Variable
import tor.optim as optim

# <<deep learning>> 4.5 example

a = tor.FloatTensor((1.0, 2.0, 3.0))
b = tor.FloatTensor((60,))
x = Variable(tor.rand(3), requires_grad=True)
sgd = optim.SGD((x,), lr=0.001)


def loss():
    va, vb = Variable(a), Variable(b)
    y = va.dot(x) - vb
    return y.dot(y) * 0.5


def sgd_small_enough(y):
    return y.data[0] < 1


def train():
    for _ in range(200000):
        sgd.zero_grad()
        y = loss()
        y.backward()
        if sgd_small_enough(y):
            break
        sgd.step()


def show():
    print('x={}'.format(x.data.numpy()))
    print('loss={}'.format(loss().data.numpy()[0]))


if __name__ == '__main__':
    train()
    show()
