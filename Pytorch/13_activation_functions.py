import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

# softmax
output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)

# sigmoid
output = torch.sigmoid(x)
s = nn.Sigmoid()
output = s(x)

# tanh
output = torch.tanh(x)
t = nn.Tanh()
output = t(x)

#relu
output = torch.relu(x)
relu = nn.ReLU()
output = relu(x)

# leaky relu
output = F.leaky_relu(x)
lrelu = nn.LeakyReLU()
output = lrelu()

# Create NN Modules
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(NeuralNet, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.sigmoid(out)
    return out

model = NeuralNet(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()

# using activation functions inside directly in forward function
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    out = torch.relu(self.linear1(x))
    out = torch.sigmoid(self.linear2(out))
    return out

model = NeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() 
