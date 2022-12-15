import torch
from torch.utils.data import Dataset, DataLoader
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
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(data_loader))

for i in range(6):
  image = images[i].movedim(0, -1)
  plt.subplot(2, 3, i+1)
  plt.imshow(image)