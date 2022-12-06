import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
print(X.shape, y.shape)

n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Model
class Model(nn.Module):
  def __init__(self, n_input_features) -> None:
    super(Model, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)
  
  def forward(self, x):
    y_pred = self.linear(x)
    y_pred_sig = torch.sigmoid(y_pred)
    return y_pred_sig

model = Model(n_features)

# parameters
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs):
  y_pred = model(X_train)
  loss = criterion(y_pred, y_train)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

with torch.no_grad():
  predictions = model(X_test)
  predictions_cls = predictions.round()
  acc = predictions_cls.eq(y_test).sum() / float(y_test.shape[0])
  print(f'accuracy: {acc.item():.4f}')