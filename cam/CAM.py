import torch
from torch import nn
from torch.nn import functional as F

from torchvision import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)
from PIL import Image

from .utils import FeatureExtractor
from .settings import *

class CAM:
    def __init__(self, model: nn.Module, dataset: str) -> None:
        self.model, self.dataset, = model, dataset
        
        self.layer_ids = []
        for name, _ in self.model.named_children():
            self.layer_ids.append(name)
        
        err_info = 'This model is incompatible with CAM!'
        if self.layer_ids[-2] != 'global_pool':
            raise ValueError(err_info)
        
        if not isinstance(
            self.model.get_submodule(self.layer_ids[-1]), 
            nn.Linear
        ):
            raise ValueError(err_info)
        
        self.feature_extractor = FeatureExtractor(
            self.model, [self.layer_ids[-3]]
        )
        
    @torch.no_grad()
    def _get_weights(self, class_idx: int):
        fc: nn.Linear = self.model.get_submodule(self.layer_ids[-1])
        return fc.weight[class_idx]

    @torch.no_grad()
    def generate_cam(self, img_pth: str, mask_rate: float = 0.5, save_pth: str = None):
        img = Image.open(img_pth).convert('RGB')
        img_np = np.array(img)
        
        if self.dataset == 'imagenette':
            tfm_resize = transforms.CenterCrop(160)
        else:
            tfm_resize = transforms.Resize(224)
            
        tfms = transforms.Compose([
            tfm_resize, 
            transforms.ToTensor()
        ])

        img_tensor: torch.Tensor = tfms(img).unsqueeze(0)
        
        y_hat = self.model(img_tensor)
        probs = F.softmax(y_hat, dim = 1)
        prob, pred_idx = probs.max().item(), probs.argmax().item()
        
        features: torch.Tensor = self.feature_extractor(img_tensor)
        featuremap = features[self.layer_ids[-3]].squeeze(0)
        
        weight = self._get_weights(pred_idx).reshape(-1, 1, 1)
        heatmap_raw: np.ndarray = (weight * featuremap).sum(dim = 0).numpy()
        
        heatmap_raw = (heatmap_raw - heatmap_raw.min()) / (heatmap_raw.max() - heatmap_raw.min())
        heatmap_raw = np.uint8(heatmap_raw * 255)
        heatmap = cv2.applyColorMap(
            cv2.resize(heatmap_raw, (img_np.shape[1], img_np.shape[0])),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  
        
        cam = np.uint8(np.array(img) * (1 - mask_rate) + heatmap * mask_rate)
        plt.imshow(cam)
        plt.title(f'{imagenette_classes[pred_idx]} {prob*100:4.2f}%', fontsize = 22)
        plt.axis('off')

        if save_pth:
            plt.savefig(save_pth, bbox_inches = 'tight')
        else:
            plt.show()
