import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import vgg19


"""
Feature extractor using pretrained VGG19 network
"""
class VGG19_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19_FeatureExtractor, self).__init__()
        self.vgg = vgg19(pretrained=True)
        
    def forward(self, x):
        out = self.vgg.features[:35](x)
        return out

    
"""
RED-CNN 
Architecture described in "Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)"
https://arxiv.org/abs/1702.00288
"""
class REDCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(REDCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 96, 5)
        self.conv2 = nn.Conv2d(96, 96, 5)
        self.conv3 = nn.Conv2d(96, 96, 5)
        self.conv4 = nn.Conv2d(96, 96, 5)
        self.conv5 = nn.Conv2d(96, 96, 5)
        self.deconv1 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv2 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv3 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv4 = nn.ConvTranspose2d(96, 96, 5)
        self.deconv5 = nn.ConvTranspose2d(96, 1, 5)
        
        
    def forward(self, x):
        res1 = x.clone()
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        
        res2 = out.clone()
        out = self.conv3(out)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.relu(out)
        
        res3 = out.clone()
        out = self.conv5(out)
        out = self.relu(out)
        
        out = self.deconv1(out)
        out = out + res3
        out = self.relu(out)
        
        out = self.deconv2(out)
        out = self.relu(out)
        
        out = self.deconv3(out)
        out = out + res2
        out = self.relu(out)
        
        out = self.deconv4(out)
        out = self.relu(out)
        
        out = self.deconv5(out)
        out = out + res1
        out = self.relu(out)
        
        return out
    

"""
Autoencoder architecture
"""
class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.MaxPool2d(2, 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    

class Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, out_channels, 3, 2, 1, 1),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out