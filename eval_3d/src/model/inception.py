import timm 
import torch
import torch.nn as nn

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class InceptionFeatureExtractor(nn.Module):
    def __init__(self,
                 model_name="inception_v3.gluon_in1k",
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        config = resolve_data_config({}, model=self.model)
        self.image_preprocessor = create_transform(**config)
    
    @torch.no_grad()
    def encode_image(self, x):
        """Takes the after global pool feature"""
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        return x
    
    @torch.no_grad()
    def encode_label_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def image_preprocessor(self, image: Image) -> torch.Tensor:
        return self.image_preprocessor(image)
    
