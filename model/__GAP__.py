import torch
from torch import nn

class GlobalAvgPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, X):
        if X.dim() != 4:
            raise ValueError(
                'The input of GAP should be batch images!'
            )
        return X.mean(dim = (-2, -1))