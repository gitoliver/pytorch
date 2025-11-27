import torch


# Backpropagation to adjust weights according to gradient of the loss function. torch.autograd.

# This is a one-layer NN.
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True) # requires_grad computes the gradients of the loss function wrt these variables
# w.requires_grad_(True) # Can do it later like this instead.
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Compute the deriviatives of the loss function wrt parameters in x and y.
loss.backward()
print(w.grad)
print(b.grad)

# By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation. However, there are some cases when we do not need to do that, for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do forward computations through the network. We can stop tracking computations by surrounding our computation code with torch.no_grad() block:
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
