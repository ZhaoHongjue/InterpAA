'''
Reference: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
'''

import torch
from torch import nn
from typing import Iterable, Dict, Callable

class VerboseExe(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.handles = []
        
        for name, module in self.model.named_children():
            module.__name__ = name
            self.handles.append(
                module.register_forward_hook(
                    lambda m, _, o: print(
                        f'{m.__name__:10}\t\t\t{o.shape}'
                    ) 
                )
            )
                
    def forward(self, X) -> None:
        self.model(X)
        for i in range(len(self.handles)):
            self.handles[i].remove()    
    
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]) -> None:
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}
        self.handles = []
        
        for layer_id in self.layers:
            layer = dict([*self.model.named_children()])[layer_id]
            self.handles.append(
                layer.register_forward_hook(self.hook_save_features(layer_id))
            )
            
    def hook_save_features(self, layer_id) -> Callable:
        def hook_fn(_, __, output):
            self._features[layer_id] = output
        return hook_fn
    
    def forward(self, X) -> Dict[str, torch.Tensor]:
        _ = self.model(X)
        for i in range(len(self.handles)):
            self.handles[i].remove() 
        return self._features
            
    