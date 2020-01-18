
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy
from torch.utils.data import Dataset
import glob
from PIL import Image

from torch.utils.data import DataLoader


class Simple_CNN(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(Simple_CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature*53*53, 50)
        self.fc2 = nn.Linear(50, output_size)
        
    def forward(self, x, verbose=False):
        #print(x.shape)
        x = self.conv1(x)
        #print('conv1size', x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        #print('maxpooling1 shape', x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        #print('conv2 size', x.shape)
        x = F.max_pool2d(x, kernel_size=2)
        #print('maxpooling2 shape', x.shape)
        x = x.view(-1, self.n_feature*53*53)
        x = self.fc1(x)
        #print('fc1 shape', x.shape)
        x = F.relu(x)
        x = self.fc2(x)
        #print('fc2', x.shape)
        x = F.log_softmax(x, dim=1)
        return x
        

        
class Cat_DOG_CNN(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(Cat_DOG_CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.conv4 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        
        self.fc1 = nn.Linear(n_feature*10*10, 50)
        self.fc2 = nn.Linear(50, output_size)
        
    def forward(self, x, verbose=False):
        #print(x.shape)
        x = self.conv1(x)
        #print('conv1size', x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        #print('maxpooling1 shape', x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        #print('conv2 size', x.shape)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        #print('conv3 size', x.shape)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv4(x)
        x = F.relu(x)
        #print('conv4 size', x.shape)
        x = F.max_pool2d(x, kernel_size=2)
        #print('maxpooling2 shape', x.shape)
        x = x.view(-1, self.n_feature*10*10)
        x = self.fc1(x)
        #print('fc1 shape', x.shape)
        x = F.relu(x)
        x = self.fc2(x)
        #print('fc2', x.shape)
        x = F.log_softmax(x, dim=1)
        return x
        