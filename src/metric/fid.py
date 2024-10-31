import numpy as np 
import scipy
from jaxtyping import Float

def fid_from_stats(m1: Float[np.ndarray, "D"], 
                   c1: Float[np.ndarray, "D D"], 
                   m2: Float[np.ndarray, "D"], 
                   c2: Float[np.ndarray, "D D"]) -> Float[np.ndarray, "1"]:
    diff = m1 - m2 
    offset = np.eye(c1.shape[0]) * 1e-6
    # prevent semi-PSD
    covmean, _ = scipy.linalg.sqrtm((c1 + offset) @ (c2 + offset), disp=False)

    # numeric error might give slight imaginary part
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(c1) + np.trace(c2) - 2 * tr_covmean