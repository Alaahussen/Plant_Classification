import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchvision import datasets
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root='train_data',
                                  transform=train_transform) 

test_data = datasets.ImageFolder(root='test_data', 
                                 transform=test_transform)


train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=32, 
                              num_workers=0, 
                              shuffle=True) 
test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=32, 
                             num_workers=0, 
                             shuffle=False) 

