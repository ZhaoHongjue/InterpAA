'''
Author: Hongjue
Email: hongjue0830@zju.edu.cn
'''

import torch
from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels) -> None:
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size = 3, padding = 1
            ))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size = 2,stride = 2))
        self.seq = nn.Sequential(*layers)
        
    def forward(self, X):
        return self.seq(X)
    
    
class VGG(nn.Module):
    def __init__(self, in_channels, out_channels, use_gap = False) -> None:
        super().__init__()
        vgg_blks = []
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        for (num_convs, mid_channels) in conv_arch:
            vgg_blks.append(VGGBlock(num_convs, in_channels, mid_channels))
            in_channels = mid_channels
            
        self.seq = nn.Sequential(
            *vgg_blks, nn.Flatten(),
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, out_channels)
        )
        
    def forward(self, X):
        return self.seq(X)
    
if __name__ == '__main__':
    X = torch.rand((128, 3, 32, 32))
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = VGG(3, 100, conv_arch)
    for layer in model.seq:
        X = layer(X)
        print(f'{layer.__class__.__name__}:\t\t\t\t{X.shape}')
