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
from torch.utils.data import DataLoader

#from PIL import Image
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
from PIL import PILLOW_VERSION
from torch.utils.data import DataLoader
import sys
from torchvision import transforms
from dataset import CatDogDataset
from model import Simple_CNN,Cat_DOG_CNN
from keras.callbacks import TensorBoard

#function to count number of parameters
# def get_n_params(model):
#     np=0
#     for p in list(model.parameters()):
#         np += p.nelement()
#     return np

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

input_size  = 224*224  # images are 224*224 pixels and has 3 channels because of RGB color
output_size = 2      # there are 2 classes---Cat and dog

image_size = (224, 224)
image_row_size = image_size[0] * image_size[1]
n_features = 4

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x:x.repeat(3,1,1))])
                                #,transforms.Normalize(mean, std)])



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


net = Cat_DOG_CNN(input_size, n_features, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)    

 
         
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

#n_features = 4 # hyperparameter
#num_flat_features(self,x)

model_cnn1 = Simple_CNN(input_size, n_features, 2)
model_cnn2 = Cat_DOG_CNN(input_size, n_features, 2)
optimizer1 = optim.SGD(model_cnn1.parameters(), lr=0.01, momentum=0.5)
optimizer2 = optim.SGD(model_cnn2.parameters(), lr=0.01, momentum=0.5)

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('CDruns/cat_dog_file')

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('Cat_Dog_images', img_grid)# try add histogram

writer.add_graph(net, images)
writer.close()

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            train_loader.dataset.classes[preds[idx]],
            probs[idx] * 100.0,
            train_loader.dataset.classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

running_loss = 0.0
for epoch in range(2):  # loop over the dataset multiple times

    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
            # ...log the running loss
          writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(train_loader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
          writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(train_loader) + i)
          running_loss = 0.0
print('Finished Training')

# 1. gets the probability predictions in a test_size x num_classes Tensor
# 2. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_preds = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)
print('class_preds=', class_preds)
# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(train_loader.dataset.classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(train_loader.dataset.classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)



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
    
        
    
