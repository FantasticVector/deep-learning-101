import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
  def __init__(self) -> None:
    xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
    self.n_samples = xy.shape[0]

    self.x_data = torch.from_numpy(xy[:, 1:])
    self.y_data = torch.from_numpy(xy[:, [0]])
  
  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]
  
  def __len__(self):
    return self.n_samples

dataset = WineDataset()

first_sample = dataset[0]
features, labels = first_sample
print(features,labels)

train_loader = DataLoader(
  dataset=dataset,
  batch_size=4,
  shuffle=True,
  num_workers=0)

dataiter = iter(train_loader)
data = next(dataiter)
features,labels = data
print('features: ', features)
print('labelss: ', labels)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
  for i, (inputs, labels) in enumerate(train_loader):
    # here training will occur
    pass
