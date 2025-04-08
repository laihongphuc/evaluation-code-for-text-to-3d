import torch
from PIL import Image
import numpy as np

from eval_3d import get_feature_extractor
from eval_3d.src.metric import mmd

def test_mmd():
    model = get_feature_extractor("clip", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    x = model.encode_image(x)
    y = torch.randn(1, 3, 224, 224)
    y = model.encode_image(y)
    # score = mmd(x, y)
    score = mmd(x, x)
    score_2 = mmd(x, y)
    print(score, score_2)
    assert score == 0
    assert score_2 != 0
