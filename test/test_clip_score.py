import torch
from PIL import Image 
import numpy as np

from src import get_feature_extractor


def test_get_feature_extractor():
    metric_name = "clip"
    model = get_feature_extractor(metric_name, pretrained=False)
    x = np.random.randint(0, 255, size=(128, 128, 3)).astype(np.uint8)
    x = Image.fromarray(x)
    x = model.image_preprocessor(x).unsqueeze(0)
    x = model.encode_image(x)
    assert x.ndim == 2

def test_get_image_preprocessor():
    metric_name = "clip"
    model = get_feature_extractor(metric_name, pretrained=False)
    x = np.random.randint(0, 255, size=(128, 128, 3)).astype(np.uint8)
    x = Image.fromarray(x)
    x = model.image_preprocessor(x)
    assert isinstance(x, torch.Tensor)
        
