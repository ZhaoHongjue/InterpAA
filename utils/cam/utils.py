'''
Reference: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
'''

import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import Iterable, Dict, Callable
from PIL import Image

from .settings import *

plt.rcParams['figure.figsize'] = (5, 5)

def _get_result(
        raw_heatmap: np.ndarray, 
        img_np: np.ndarray, 
        mask_rate: float = 0.4
    ) -> np.ndarray:
        raw_max, raw_min = raw_heatmap.max(), raw_heatmap.min()
        raw_heatmap = (raw_heatmap - raw_min) / (raw_max - raw_min)
        raw_heatmap = np.uint8(raw_heatmap * 255)
        heatmap = cv2.applyColorMap(
            cv2.resize(raw_heatmap, (img_np.shape[1], img_np.shape[0])),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  
        
        cam_img_np = np.uint8(img_np * (1 - mask_rate) + heatmap * mask_rate)
        return cam_img_np

def plot_cam_img(
    cam_img_np: np.ndarray, dataset: str, class_idx: int, 
    prob: float, save_pth: str = None
) -> None:
    plt.clf()
    class_name = class_names[dataset][class_idx]
    plt.imshow(cam_img_np)
    if 0 <= prob <= 1:
        prob *= 100
    plt.title(f'{class_name} {prob:4.2f}%', fontsize = 22)
    plt.axis('off')
    
    if save_pth:
        plt.savefig(save_pth, bbox_inches = 'tight', pad_inches = 0.03)
    else:
        plt.show()
    

class VerboseExe:
    def __init__(self, model: nn.Module) -> None:
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
                
    def __call__(self, X) -> None:
        self.model(X)
        for i in range(len(self.handles)):
            self.handles[i].remove()    
    
class FeatureExtractor:
    def __init__(self, model: nn.Module, layers: Iterable[str]) -> None:
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
    
    def __call__(self, X) -> Dict[str, torch.Tensor]:
        _ = self.model(X)
        for i in range(len(self.handles)):
            self.handles[i].remove() 
        return self._features
            
