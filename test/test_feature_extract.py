import torch
from PIL import Image 
import numpy as np

from eval_3d import get_feature_extractor


def test_get_feature_extractor():
    metric_name = "fid"
    model = get_feature_extractor(metric_name, pretrained=False)
    x = torch.randn(1, 3, 299, 299)
    x = model.encode_image(x)
    assert x.ndim == 2

def test_get_image_preprocessor():
    metric_name = "fid"
    model = get_feature_extractor(metric_name, pretrained=False)
    x = np.random.randint(0, 255, size=(128, 128, 3)).astype(np.uint8)
    x = Image.fromarray(x)
    x = model.image_preprocessor(x)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 299, 299)
        
