import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]],dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([[5]], dtype=torch.float32)
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, input_size)
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
  y_pred = model(X)
  l = loss(y_pred, Y)
  l.backward()
  optimizer.step()
  optimizer.zero_grad()

print(model(X_test).item())


