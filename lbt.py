import numpy as np
from cued_sf2_lab.dct import colxfm

from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dct import dct_ii

from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from functools import lru_cache

entropy = lambda X: bpp(X) * np.prod(X.size)
energy = lambda X: np.sum(X ** 2.0)

@lru_cache
def C(N):
    return dct_ii(N)

@lru_cache
def lbt_encode(X, s=None, N=8):
    t = np.s_[N//2:-N//2]
    Pf = pot_ii(N, s)[0]
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

    Y = colxfm(colxfm(Xp, C(N)).T, C[N]).T
    return Y

def lbt_decode(Y, s=None, N=8):
    t = np.s_[N//2:-N//2]
    Pr = pot_ii(N, s)[1]
    Z = colxfm(colxfm(Y.T, C(N).T).T, C(N).T)
    Zp = Z.copy()  #copy the non-transformed edges directly from Z
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    return Zp

def lbt_std(X, qnum, s, N=8):
    Y = lbt_encode(X, s=s, N=N)
    Zp = lbt_decode(quantise(Y, qnum), s=s, N=N)
    return np.std(X-Zp)

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