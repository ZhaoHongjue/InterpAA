import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class GradCAM(BaseCAM):
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: str, 
        fc_layer: str = None, 
        use_relu: bool = False, 
        use_cuda: bool = True
    ) -> None:
        super().__init__(
            model, target_layer, fc_layer, use_relu, use_cuda
        )
        
    def _get_weights(self, img_tensor: torch.Tensor) -> torch.Tensor:
        grad_outs = []
        handle = self.model.layer4.register_full_backward_hook(
            lambda _, __, go: grad_outs.append(go)
        )
        
        self.model.zero_grad()
        logits = self.model(img_tensor)
        idx = logits.argmax(dim = 1)
        logits[:, idx].backward()
        
        grads = grad_outs[0][0].squeeze(0)
        gap = nn.AdaptiveAvgPool2d((1, 1))
        handle.remove()
        return gap(grads)



        