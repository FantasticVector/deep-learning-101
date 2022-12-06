import torch

x = torch.empty(1) # scaler
y = torch.empty(3) # 1D
z = torch.empty(2, 3) # 2D
x = torch.empty(2, 2, 2) # 3D
# random matrix of 5x3
x = torch.rand(5, 3)
x = torch.zeros(5, 3) # zeros matrix
x = torch.ones(5, 3)# ones matrix
x = torch.tensor([5, 3]) # from custom data
x = torch.ones(5, 3, dtype=torch.float16)
x = torch.tensor([5.5, 3], requires_grad=True) # cal gradients

x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x + y
z = x - y
z = x * y
z = torch.add(x, y) # y.add_(x) inplace y
z = torch.sub(x, y) 
z = torch.mul(x, y)
z = torch.div(x, y)
 
# slicing
x = torch.rand(5, 3)
print(x[:, 0]) # all rows and 0 col
print(x[1, :]) # 1 row and all cols
print(x[1, 1]) # [1, 1] element: returns tensor(0.142)
print(x[1, 1].item()) # return actual value

x = torch.randn(4, 4)
y = x.view(16) # changing dimensions
z = x.view(-1, 8)
print(y.size(), z.size())

# to numpy
a = torch.ones(5)
b = a.numpy()
print(b)
# from numpy to torch
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# run all operations only on gpu
if torch.cuda.is_available():
  device = torch.device('cuda')
  y = torch.ones_like(x, device=device)
  x = x.to(device)
  