'''
Author: Hongjue
Email: hongjue0830@zju.edu.cn
'''

import torch
from torch import nn

class NiNBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, 
        kernel_size: int = 3, stride: int = 1, padding: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 1)
        self.relu3 = nn.ReLU()
        
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
    
class NiN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_gap: bool = True) -> None:
        super().__init__()
        self.nin_blk1 = NiNBlock(in_channels, 64, kernel_size = 3, padding = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.nin_blk2 = NiNBlock(64, 128, kernel_size = 3, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.nin_blk3 = NiNBlock(128, 256, kernel_size = 3, padding = 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 1)
        self.dropout1 = nn.Dropout(0.5)
        
        self.nin_blk4 = NiNBlock(256, 512, kernel_size = 3, padding = 1)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 1)
        self.dropout2 = nn.Dropout(0.5)
        
        self.nin_blk5 = NiNBlock(512, out_channels, kernel_size = 3, padding = 1)
        self.maxpool5 = nn.MaxPool2d(kernel_size = 2, stride = 1)
        
        self.dropout = nn.Dropout(0.5)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X