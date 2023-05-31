import numpy as np
from functools import lru_cache
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dct import regroup

entropy = lambda X: bpp(X) * np.prod(X.size)
energy = lambda X: np.sum(X ** 2.0)

@lru_cache
def C(N):
    return dct_ii(N)

def dct_encode(X, N=8):
    return colxfm(colxfm(X, C(N)).T, C(N)).T

def dct_decode(Y, N=8):
    return colxfm(colxfm(Y.T, C(N).T).T, C(N).T)

def dct_std(X, qnum, N=8):
    Y = dct_encode(X, N=N)
    Yq = quantise(Y, qnum)
    Z = dct_decode(Yq, N=N)
    return np.std(X-Z)

def dctbpp(Yr, N):
    total_bits = 0
    subimage_shape = tuple([d // N for d in Yr.shape])
    for horiz_freq in range(N):
        for vert_freq in range(N):
            subimage = Yr[
                subimage_shape[0] * horiz_freq:subimage_shape[0] * horiz_freq + subimage_shape[0],
                subimage_shape[1] * vert_freq:subimage_shape[1] * vert_freq + subimage_shape[1]
            ]
            total_bits += bpp(subimage) * subimage.size
    return total_bits