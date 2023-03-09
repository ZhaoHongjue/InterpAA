'''
Author: Hongjue
Email: hongjue0830@zju.edu.cn
'''

import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, num_channels: int,
                 use_1x1conv: bool = False, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, num_channels,
            kernel_size = 3, padding = 1, stride = stride
        )
        self.conv2 = nn.Conv2d(
            num_channels, num_channels,
            kernel_size = 1, stride = stride
        )
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels, num_channels,
                kernel_size = 1, stride = stride
            )
            
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if hasattr(self, 'conv3'):
            X = self.conv3(X)
        Y = Y + X
        return F.relu(Y)
    
class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_gap: bool = True) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.res_idx = 1
        self.add_res_blk(64, 64, num_res = 2, first_blk = True)
        self.add_res_blk(64, 128, num_res = 2)
        self.add_res_blk(128, 256, num_res = 2)
        self.add_res_blk(256, 512, num_res = 2)
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(512, out_channels)
        
    def add_res_blk(self, in_channels, num_channels, num_res, first_blk = False):
        for i in range(num_res):
            if i == 0 and not first_blk:
                self.add_module(
                    f'res_blk{self.res_idx}', 
                    ResBlock(
                        in_channels, num_channels, 
                        use_1x1conv = True, stride = 1
                    )
                )
            else:
                self.add_module(
                    f'res_blk{self.res_idx}',
                    ResBlock(num_channels, num_channels)
                )
            self.res_idx += 1
    
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X