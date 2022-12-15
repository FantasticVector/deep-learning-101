import os
import shutil
path = "./data/dogs-vs-cats/train/"
for folder in ['cat', 'dog']:
  os.makedirs(path+folder, exist_ok=True)

for filename in os.listdir(path):
  folder = filename.split('.')[0]
  shutil.move(path+filename, path+folder)