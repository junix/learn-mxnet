import tor
import tor.nn as nn
from tor.autograd import Variable

# ======================== Basic autograd example 2 =======================#
# Create tensors.
x = Variable(tor.randn(5, 3))
y = Variable(tor.randn(5, 2))

# Build a linear layer.
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# Build Loss and Optimizer.
criterion = nn.MSELoss()
optimizer = tor.optim.SGD(linear.parameters(), lr=0.01)

# Forward propagation.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.data[0])

# Backpropagation.
loss.backward()

# Print out the gradients.
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step Optimization (gradient descent).
optimizer.step()

# You can also do optimization at the low level as shown below.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after optimization.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.data[0])