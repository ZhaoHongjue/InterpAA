import torch
from torch import nn

class VerboseExe(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        
        for name, module in self.model.named_children():
            module.__name__ = name
            module.register_forward_hook(
                lambda m, _, o: print(
                    f'{m.__name__:10}\t\t\t{o.shape}'
                ) 
            )
                
    def forward(self, X):
        return self.model(X)