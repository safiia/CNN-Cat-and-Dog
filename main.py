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
#from PIL import Image
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
from PIL import PILLOW_VERSION
from torch.utils.data import DataLoader
import sys

from dataset import CatDogDataset
from model import Simple_CNN,Cat_DOG_CNN


#function to count number of parameters
# def get_n_params(model):
#     np=0
#     for p in list(model.parameters()):
#         np += p.nelement()
#     return np

input_size  = 224*224  # images are 224*224 pixels and has 3 channels because of RGB color
output_size = 2      # there are 2 classes---Cat and dog

image_size = (224, 224)
image_row_size = image_size[0] * image_size[1]


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x:x.repeat(3,1,1)),
                                transforms.Normalize(mean, std)])



path    = './Cat_Dog_data/train'
path1 = './Cat_Dog_data/test'
train = CatDogDataset(path, transform=transform)
test = CatDogDataset(path1, transform=transform)

shuffle     = True
batch_size  = 64
num_workers = 1

train_loader  = DataLoader(dataset=train, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)
test_loader  = DataLoader(dataset=test, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
            
accuracy_list = []

def train(epoch, model,optimizer):# perm=torch.arange(0, 80656).long()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # permute pixels
        #data = data.view(-1, 224*224)
        #data = data[:, perm]
        #data = data.view(-1, 3, 224,224)
        
        optimizer.zero_grad()
        output = model(data)
        #print('target shape', target)
        #print('output', output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, perm=torch.arange(0, 224*224*3).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # permute pixels
        #data = data.view(-1, 224*224)
        #data = data[:, perm]
       # data = data.view(-1, 3, 224,224)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    

# Training settings 

n_features = 4 # hyperparameter
#num_flat_features(self,x)

model_cnn1 = Simple_CNN(input_size, n_features, 2)
model_cnn2 = Cat_DOG_CNN(input_size, n_features, 2)
optimizer1 = optim.SGD(model_cnn1.parameters(), lr=0.01, momentum=0.5)
optimizer2 = optim.SGD(model_cnn2.parameters(), lr=0.01, momentum=0.5)

#print('Number of parameters: {}'.format(get_n_params(model_cnn)))
def model_one():
    
    
    for epoch in range(0, 1):
        train(epoch, model_cnn1,optimizer1)
        test(model_cnn1)

def model_two():
    
    
    for epoch in range(0, 1):
        train(epoch, model_cnn2,optimizer2)
        test(model_cnn2)  
    
if  __name__ =="__main__":
    if len(sys.argv[1]) >1 and (sys.argv[1]== 'model_one'):
        model_one()
    else: 
        model_two()
    
        
    