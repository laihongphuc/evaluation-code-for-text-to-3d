import numpy as np 
import torch 
import scipy
from jaxtyping import Float


def entropy(
    probs: Float[np.ndarray, "D"],
) -> Float[np.ndarray, "1"]:
    return scipy.stats.entropy(probs)

def inception_variety_from_probs(
    probs: Float[torch.Tensor, "B D"], 
) -> Float[np.ndarray, "1"]:
    probs = probs.cpu().numpy()
    return entropy(np.mean(probs, axis=0))

def inception_gain_from_probs(
    probs: Float[torch.Tensor, "B D"],
) -> Float[np.ndarray, "1"]:
    probs = probs.cpu().numpy()
    iv = entropy(np.mean(probs, axis=0))
    iq = - np.sum(probs * np.log(probs), axis=1).mean()
    return (iv - iq) / (iq + 1e-6)