import numpy as np 
import scipy
from jaxtyping import Float

def fid_from_stats(m1: Float[np.ndarray, "D"], 
                   c1: Float[np.ndarray, "D D"], 
                   m2: Float[np.ndarray, "D"], 
                   c2: Float[np.ndarray, "D D"]) -> Float[np.ndarray, "1"]:
    diff = m1 - m2 
    offset = np.eye(c1.shape[0]) * 1e-6
    covmean, _ = scipy.linalg.sqrtm((c1 + offset) @ (c2 + offset), disp=False)
    tr_covmean = np.trace(covmean)
    breakpoint()
    return diff.dot(diff) + np.trace(c1) + np.trace(c2) - 2 * tr_covmean