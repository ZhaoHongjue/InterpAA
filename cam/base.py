import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)
from PIL import Image

from typing import List
from abc import abstractmethod

from .utils import FeatureExtractor, _get_result
from .settings import *

class BaseCAM:
    def __init__(
        self, model: nn.Module, 
        target_layer: str,
        fc_layer: str = None,
        use_relu: bool = False,
        use_cuda: bool = True
    ) -> None:
        self.model = model
        self.use_relu = use_relu
        layer_names = self._get_layer_names()
        
        if target_layer not in layer_names:
            raise AttributeError(
                f'Model has no attribute `{target_layer}`!'
            )
        
        if fc_layer:
            if fc_layer not in layer_names:
                raise AttributeError(
                    f'Model has no attribute `{fc_layer}`'
                )
            if not isinstance(
                self.model.get_submodule(fc_layer), 
                nn.Linear
            ):
                raise ValueError(
                    f'{fc_layer} is not a `nn.Linear` instance!'
                )
        
        if use_cuda:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)
        self.feature_extractor = FeatureExtractor(
            self.model, [target_layer]
        )
        
        self.fc_layer = fc_layer
        self.target_layer = target_layer
    
    def __call__(
        self, img: Image.Image, mask_rate: float = 0.4, 
        centercrop: int = None, resize: int = None,
    ) -> Image.Image:
        img_np = np.array(img)
        
        tfm_lst = []
        if centercrop:
            tfm_lst.append(transforms.CenterCrop(centercrop))
        if resize:
            tfm_lst.append(transforms.Resize(resize))
        tfm_lst.append(transforms.ToTensor())
        tfms = transforms.Compose(tfm_lst)
        img_tensor: torch.Tensor = tfms(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        weights: torch.Tensor = self._get_weights(img_tensor)
        featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
        raw_heatmap: torch.Tensor = (weights * featuremaps).sum(dim = 0)
        if self.use_relu:
            raw_heatmap = F.relu(raw_heatmap)
        raw_heatmap = raw_heatmap.cpu().numpy()
        return _get_result(raw_heatmap, img_np, mask_rate)
    
    @torch.no_grad()
    def _get_feature_maps(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(img_tensor)[self.target_layer]
    
    @abstractmethod
    def _get_weights(self, img_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def _get_layer_names(self) -> List[str]:
        layer_names = []
        for name, _ in self.model.named_children():
            layer_names.append(name)
        return layer_names
    
    