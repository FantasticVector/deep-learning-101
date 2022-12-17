import torch
import torch.nn as nn

class Perceptron(nn.Module):
  def __init__(self, input_dim) -> None:
    super().__init__()
    self.fc = nn.Linear(input_dim, 1)
  
  def forward(self, x):
    return torch.sigmoid(self.fc(x)).squeeze()