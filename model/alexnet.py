import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size = 3, 
                      stride = 2, padding = 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(96, 256, kernel_size = 5, padding = 2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, out_channels)
        )
        
    def forward(self, X):
        return self.seq(X)