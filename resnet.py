# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:00:47 2020

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#class to define the Bulding Block in ResNet (Figure 5 left in original paper)
class BuildingBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BuildingBlock, self).__init__()   #constructor
        #first conv layer
        #in_planes is input size, planes is output 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        #batch normalization layer
        self.bn1 = nn.BatchNorm2d(planes)
        #second conv layer
        #planes is input and second planes is output
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        #batch normalization layer for conv2
        self.bn2 = nn.BatchNorm2d(planes)
        
        
        #define the shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                                          nn.BatchNorm2d(self.expansion * planes))
            
        
    #define the forward pass in the network    
    def forward(self, x):
        #conv1 --> bn1 --> relu --> conv2 --> bn2 --> shortcut (input x from conv1 is passed here) --> relu --> output
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = output + self.shortcut(x)
        output = F.relu(output)
        return output
        
        
#class to define the Bulding Block in ResNet (Figure 5 right in original paper)    
class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
         #define the shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                                          nn.BatchNorm2d(self.expansion * planes))
            
    
    #define forward pass in BottleNeck building block    
    def forward(self, x):
        #conv1 --> bn1 --> relu --> conv2 --> bn2 --> relu --> conv3 --> bn3 --> shortcut (input x from conv1 is passed here) --> relu --> output   
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output = output + self.shortcut(x)
        output = F.relu(output)
        return output
        
 
#class to represent the whole architecture of ResNet
class ResNet(nn.Module):
    #num_classes = 10 because of Cifar 10 only has 10 classes
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        
        #the four layers where each layer can contain a 
        #BuildingBlock (shortcut in every 2 layers) OR
        #BottleNeck block (shortcut in every 3 layers)
        self.layer1 = self.create_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self.create_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self.create_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self.create_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
    
    def create_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        #create an empty list to accumulate / append each layer inside it
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
    
    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output
    
#define ResNet 101 layer
#use BottleNeck blocks
#layer1 - 3, layer2 - 4, layer3 - 23, layer4 - 3
def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes = 10)


#run this to check the network architecture 
#net = ResNet101()
#print(net)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        