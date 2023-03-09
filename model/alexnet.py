import torch
from torch import nn

from .__GAP__ import GlobalAvgPool2d

class AlexNet(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        use_gap = True
    ) -> None:
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size = 3, padding = 1
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, padding = 1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, padding = 1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.relu5 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, padding = 1)
        
        if use_gap:
            self.gap = GlobalAvgPool2d()
            self.fc = nn.Linear(128, out_channels)
        else:
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(128*4*4, out_channels)
        self.register_params()
    
    def register_params(self):
        def initialize(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        self.apply(initialize)
        
    def forward(self, X):
        Y = self.maxpool1(self.relu1(self.conv1(X)))
        # Y = self.maxpool2(self.relu2(self.conv2(Y)))
        Y = self.relu3(self.conv3(Y))
        Y = self.relu4(self.conv4(Y))
        Y = self.maxpool3(self.relu5(self.conv5(Y)))
        if hasattr(self, 'gap'):
            Y = self.gap(Y)
        else:
            Y = self.flatten(Y)
        Y = self.fc(Y)
        return Y    