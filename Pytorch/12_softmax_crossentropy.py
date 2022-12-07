import torch
import torch.nn as nn
import numpy as np


def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 1.0])
outputs = softmax(x)
print('Softmax output: ', outputs)

x = torch.tensor([2.0, 1.0, 1.0])
outputs = torch.softmax(x, dim=0)
print('softmax torch: ',outputs)

def cross_entropy(actual, predicted):
  EPS = 1e-15
  predicted = np.clip(predicted, EPS, 1-EPS)
  loss = -np.sum(actual * np.log(predicted))
  return loss

Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print("loss1: ", l1)
print("loss2: ", l2)

# nn.LogSoftmax + nn.NLLLoss (negative log liklihood loss)
loss = nn.CrossEntropyLoss()
# loss(input, target)


Y = torch.tensor([0])
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1, l2)

# get predictions:
_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad, 1)
print(prediction1, prediction2)


Y = torch.tensor([2, 0, 1])
Y_pred_good = torch.tensor(
  [[2.0, 1.0, 0.1],
  [1.2, 0.1, 0.3],
  [0.3, 2.2, 0.2]])

l1 = loss(Y_pred_good, Y)
print(l1)
predictions = torch.max(Y_pred_good, 1)
print(predictions)

# Binary Classification
class NeuralNet1(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(NeuralNet1, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    y_pred = torch.sigmoid(out)
    return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()

# Binary Classification
class NeuralNet2(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet1, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    # no need to pass through softmax
    return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # applies softmax