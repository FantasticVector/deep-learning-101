import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# to tensor
X = torch.from_numpy(X_numpy.astype(np.float32)) # shape: [100 1]
y = torch.from_numpy(y_numpy.astype(np.float32)) # shape: [100]
y = y.view(y.shape[0], 1) #shape: [100, 1]

n_samples, n_features = X.shape
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 100

# training
for epoch in range(num_epochs):
  y_predicted = model(X)
  loss = criterion(y_predicted
  , y)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()