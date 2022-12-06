import torch
x = torch.randn(3, requires_grad = True)
y = x * 2
print(x)
print(y)
print(y.grad_fn)

z = y * y * 3
print(z)
z = z.mean()
print(z)
z.backward()
print(x.grad)

a = torch.randn(2, 2)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = ((a*3)/(a-1))
print(b.grad_fn)

# detach
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)

with torch.no_grad():
  print((x**2).requires_grad)

weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
  model_output = (weights*3).sum()
  model_output.backward()
  print(weights.grad)

  with torch.no_grad():
    weights -= 0.1 * weights.grad
  
  weights.grad.zero_()
print(weights)
print(model_output)

weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD([weights], lr=0.1)
for epoch in range(3):
  outputs = (weights*3).sum()
  outputs.backward()
  optimizer.step()
  optimizer.zero_grad()
print(weights)