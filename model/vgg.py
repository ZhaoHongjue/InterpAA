'''
Author: Hongjue
Email: hongjue0830@zju.edu.cn
'''

import torch
from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, num_convs: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size = 3, padding = 1
            ))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.seq = nn.Sequential(*layers)
        
    def forward(self, X):
        return self.seq(X)
    
    
class VGG(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_arch, use_gap: bool = False) -> None:
        super().__init__()
        vgg_blks = []
        
        for idx, (num_convs, mid_channels) in enumerate(conv_arch):
            self.add_module(f'vgg_blk{idx+1}', VGGBlock(num_convs, in_channels, mid_channels))
            in_channels = mid_channels
            
        if use_gap:
            self.gap = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.fc = nn.Linear(conv_arch[-1][1], out_channels)
        else:
            self.flatten = nn.Flatten()
            self.fc = nn.LazyLinear(out_channels)
        
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
    
if __name__ == '__main__':
    X = torch.rand((128, 3, 32, 32))
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = VGG(3, 100, conv_arch)
    for layer in model.seq:
        X = layer(X)
        print(f'{layer.__class__.__name__}:\t\t\t\t{X.shape}')
