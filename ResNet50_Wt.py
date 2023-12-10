# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:51:19 2023

@author: Brian
"""
import os
import torch
from torch import nn
import torchvision
from MammoData import get_df, get_data
#import Mammo_VGG_Models
#import imagenet_pretraining
#from imagenet_pretraining import parse_args
#import argparse
from inflate import inflate_model
from multiprocessing import Process, freeze_support
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F


# Initializing preprocessing at standard 224x224 resolution
weights = ResNet50_Weights.DEFAULT
preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=False)


class ResNet50_Weighted(nn.Module):
    def __init__(self):
        super(ResNet50_Weighted, self).__init__()
        self._model = resnet50(weights=weights)
        
        # Disable gradient computation for the pre-trained ResNet50
        for param in self._model.parameters():
            param.requires_grad = False
        
        self.sequential = nn.Sequential(
        nn.Linear(3, 12, bias=False),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(12, 3, bias=False))
        
    def forward(self, x, additionalWeights=None):
        # Apply global average pooling to get a fixed-size representation
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.sequential(x)
        return self._model(x.unsqueeze(-1).unsqueeze(-1))  # Add dimensions for H and W


class LinearLayer1(nn.Module):
    def __init__(self):
        super(LinearLayer1, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(1000, 500, bias=False),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(500, 250, bias=False)) 
        
    def forward(self, x, additionalWeights=None):
        logits = self.linear_relu_stack(x)
        return (logits)
        
class LinearLayer2(nn.Module):
    def __init__(self):
        super(LinearLayer2, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(250, 500, bias=False),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(500, 250, bias=False))

    def forward(self, x, additionalWeights=None):
        logits = self.linear_relu_stack(x)
        return (logits)
    

class ChainedModel(nn.Module):
    def __init__(self):
        super(ChainedModel, self).__init__()
        self.module1 = ResNet50_Weighted()
        self.module2 = LinearLayer1()
        self.module3 = LinearLayer2()
                
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(250, 500, bias=False),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(500, 224, bias=False))

    def forward(self, x, additionalWeights=None):
        # Use additional_weights in the forward pass
        if additionalWeights is not None:
            x = x * additionalWeights
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.linear_relu_stack(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
    
        
model = ChainedModel()
#print('Model:' f'{model}''\n')
print("Total params:",sum([torch.prod(torch.tensor(i.shape)) for i in model.parameters()]))

def train(model, train_dataloader, optimizer, print_freq=10):
    model.train()

    train_loss = 0
    total_samples = 0
    
    for batch_index, (data, target) in enumerate(train_dataloader):
        
        optimizer.zero_grad()
        
        # Apply weight transform to the input image
        pre_processed_data = preprocess(data)
        
        # pass the data through the model to get the model output
        output = model(pre_processed_data)

        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item() * pre_processed_data.shape[0]
        total_samples += len(pre_processed_data)
        
        
        if not (batch_index % print_freq):
            # print the current training loss of the model.
            print(f"Batch: {batch_index + 1} | Training Loss: {train_loss/total_samples:.5f}")

    #average_loss = train_loss / total_samples

    return train_loss / len(train_dataloader.dataset)

def test(model, test_dataloader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        # for every batch of data and labels in the test dataloader
        for batch_index, (data, target) in enumerate(test_dataloader):
            
            # Apply weight transform to the input image
            pre_processed_data = preprocess(data)
            
            # pass the data through the model to get the model output
            output = model(pre_processed_data)
            
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_accuracy = correct / len(test_dataloader.dataset)
    
    return test_loss, test_accuracy

import matplotlib.pyplot as plt

def train_model(model, train_dataloader, test_dataloader, optimizer, num_epochs):
    x = []
    y = []
    epochs = []
    for i in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer)
        test_loss, test_accuracy = test(model, test_dataloader)

        print(
            f'Epoch: {i + 1} | Train loss: {train_loss:.5f} |',
            f'Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.5f}'
        )
        x.append(test_accuracy)
        y.append(test_loss)
        epochs.append(i)
    return [x, y, epochs]


train_data, test_data, additionalWeights = get_data()    
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=12, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=12, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 100

if __name__ == '__main__':
    freeze_support()
    x, y, epochs = train_model(model, train_dataloader, test_dataloader, optimizer, num_epochs)
    
    #Plot test accuracy against epochs
    plt.plot(epochs, x)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Epochs')
    plt.grid(True)
    plt.show()
    
    #Plot train lossees against epochs
    plt.plot(epochs, y)
    plt.xlabel('Epochs')
    plt.ylabel('Train Losses')
    plt.title('Train Losses vs Epochs')
    plt.grid(True)
    plt.show()