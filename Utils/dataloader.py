import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import torchvision
import os
from torchvision import transforms
import matplotlib.pyplot as plt
torch.manual_seed(17)

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
data_path = "D:\Courses\ML\deep-learning-101\data\dogs-vs-cats"

dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)

train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])

train_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
