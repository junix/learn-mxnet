import torch

a = torch.Tensor([[1, -2]])
b = torch.Tensor([[40]])
x = torch.Tensor([[1], [-2]])


def f():
    return (a.mm(x) - b).norm(p=2)**2

# grad
def grad():
    return a.t().mm(a.mm(x) - b)


while True:
    g = grad()
    if g.norm() < 0.1:
        break
    x.sub_(1e-4 * g)

print(x)
print(f())
