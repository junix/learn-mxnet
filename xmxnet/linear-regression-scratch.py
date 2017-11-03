from xmxnet import ndarray as nd
from xmxnet import autograd
import random

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

batch_size = 10


def data_iter():
    # 产生一个随机索引
    indexes = list(range(num_examples))
    random.shuffle(indexes)
    for beg in range(0, num_examples, batch_size):
        end = min(beg + batch_size, num_examples)
        seg = nd.array(indexes[beg:end])
        yield nd.take(X, seg), nd.take(y, seg)


w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]

for param in params:
    param.attach_grad()


def net(X):
    # batch_size x num . num x 1 => batch_size x 1
    return nd.dot(X, w) + b


def SGD(params, learning_rate):
    for p in params:
        p[:] = p - learning_rate * p.grad


def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免自动广播
    return (yhat - y.reshape(yhat.shape)) ** 2


learning_rate = .001


def step(epoch):
    total_loss = 0
    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (epoch, total_loss / num_examples))


def train():
    epochs = 10
    for e in range(epochs):
        step(e)


def print_result():
    print('w={},true_w={}'.format(w.reshape((1, 2))[0].asnumpy(), true_w))
    print('b={},true_b={}'.format(b.asscalar(), true_b))


if __name__ == '__main__':
    train()
    print_result()
