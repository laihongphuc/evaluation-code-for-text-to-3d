import torch 
import numpy as np 
from jaxtyping import Float

@torch.no_grad()
def clip_score_compute(image_features: Float[torch.Tensor, "N D"], 
               text_features: Float[torch.Tensor, "1 D"], 
               normalized: str = False) -> Float[torch.Tensor, "1"]:
    logit_scale = 100
    score_acc = 0.
    if normalized is False:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = image_features * text_features
    similarity = similarity.sum() * logit_scale
    return similarity
