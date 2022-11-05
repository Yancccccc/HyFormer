import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.functional as F

class mlpcnn(nn.Module):
    def __init__(self, num_features=8,dim=64,num_classes=9):
        super(mlpcnn, self).__init__()
        self.conv0 = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                               bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.num_features = num_features
        # Append new layers
        n_fc1 = 256
        self.feat_spe = nn.Linear(self.num_features,n_fc1)
        self.classifier = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = torch.flatten(x,start_dim=1,end_dim=2)  # (4,2) → 8
        x = self.feat_spe(x)   # 8→256
        x = x.reshape([x.size(0),16,4,4]) # batch,16,4,4
        x = self.conv0(x)
        x_res = x
        x = self.conv1(x)
        x = self.relu(x + x_res)
        x = self.avgpool(x)
        x_res = x
        x = self.conv2(x)
        x = self.relu(x + x_res)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        return self.classifier(x)
