import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cpu')

# Hyper-Parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Dataset
train_dataset = torchvision.datasets.MNIST(
  root='./data',
  train=True,
  transform=transforms.ToTensor(),
  download=True,
)
test_dataset =torchvision.datasets.MNIST(
  root='./data',
  train=False,
  transform=transforms.ToTensor(),
)

# Data Loader
train_loader = torch.utils.data.DataLoader(
  dataset = train_dataset,
  batch_size = batch_size,
  shuffle=True)

test_loader = torch.utils.data.DataLoader(
  dataset = test_dataset,
  batch_size = batch_size,
  shuffle = False
)

examples = iter(test_loader)
example_data, example_targets = next(examples)

for i in range(6):
  plt.subplot(2, 3, i+1)
  plt.imshow(example_data[i][0], cmap='gray')
plt.show()
print(example_data.size())


class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.input_size = input_size
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    return out

model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.reshape(-1, 28*28)

    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (i + 1) % 100 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for images, labels in test_loader:
    images = images.reshape(-1, 28*28)
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()
  
  acc = 100.0 * n_correct / n_samples
  print('Accuracy: ', acc)