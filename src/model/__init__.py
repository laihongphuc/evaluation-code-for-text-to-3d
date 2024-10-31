from typing import Tuple 

import torch 
import torch.nn as nn
import numpy as np
from jaxtyping import Float
from torch.utils.data import DataLoader

from .inception import InceptionFeatureExtractor
from .clip import ClipFeatureExtractor

def get_feature_extractor(metric_name, pretrained=False):
    if metric_name == "clip":
        return ClipFeatureExtractor(pretrained=pretrained)
    elif metric_name == "fid":
        return InceptionFeatureExtractor(pretrained=pretrained)
    else:
        raise NotImplementedError(f"Don't support metric {metric_name}")
    

@torch.no_grad()
def get_static_from_dataloader(model: nn.Module,
                               dataloader: DataLoader,
                               device: str) -> Tuple[Float[np.ndarray, "D"], Float[np.ndarray, "D D"]]:
    model.eval()
    model.to(device)
    total_features = []
    for img in dataloader:
        img = img.to(device)
        features = model.encode_image(img)
        total_features.append(features.cpu().numpy())
    total_features = np.concatenate(total_features, axis=0)
    mean = np.mean(total_features, axis=0)
    cov = np.cov(total_features.T)
    return mean, cov


@torch.no_grad()
def get_feature_from_dataloader(model: nn.Module,
                                dataloader: DataLoader,
                                device: str) -> torch.Tensor:
    model.eval()
    model.to(device)
    total_features = []
    for img in dataloader:
        img = img.to(device)
        features = model.encode_image(img)
        total_features.append(features)
    total_features = torch.cat(total_features, dim=0)
    return total_features