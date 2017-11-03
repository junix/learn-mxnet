import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.nn import MSELoss
from data.linear_reg import EFFORT_DATA as DATA


class Model:
    def __init__(self):
        # self.w = Variable(torch.rand(2, 1), requires_grad=True)
        self.w = Variable(torch.Tensor([[0.26193029], [0.97255689]]), requires_grad=True)
        # self.b = Variable(torch.rand(1, 1), requires_grad=True)
        self.b = Variable(torch.Tensor([[-13.84867859]]), requires_grad=True)

    def parameters(self):
        return self.w, self.b

    def input(self, x):
        x = Variable(x)
        return x.mm(self.w) + self.b

    def show(self):
        print('a={},b={}'.format(self.w.data.numpy(), self.b.data.numpy()))


def data():
    t = torch.Tensor(DATA)
    # t = torch.Tensor([
    #     [46, 0, 1],
    #     [74, 0, 10]])
    return t[:, :-1], t[:, -1:]


def loss(ys, yshat):
    l0 = yshat - ys
    l1 = l0.pow(2)
    y = l1.mean()
    return y


def train(model):
    optimizer = SGD(model.parameters(), lr=1e-5)
    for epoch in range(10000000):
        # model.show()
        xs, ys = data()
        optimizer.zero_grad()
        yshat = model.input(xs)
        l = loss(Variable(ys), yshat)
        if l.data[0] < 34.715:
            break
        l.backward()
        # print('w.grad={}'.format(model.w.grad.data.numpy()))
        print('loss={}'.format(l.data[0]))
        optimizer.step()


if __name__ == '__main__':
    model = Model()
    train(model)
    model.show()
