from functools import lru_cache
from typing import Tuple
import numpy as np

from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dwt import dwt, idwt

entropy = lambda X: bpp(X) * np.prod(X.size)
energy = lambda X: np.sum(X ** 2.0)


def nlevdwt(X, n):
    m = X.shape[0]
    Y = X.copy()
    for _ in range(n):
        Y[:m, :m] = dwt(Y[:m, :m])
        m = m // 2
    return Y


def nlevidwt(Y, n):
    m = Y.shape[0] // (2 ** (n - 1))
    X = Y.copy()
    for _ in range(n):
        
        X[:m, :m] = idwt(X[:m, :m])
        m = m * 2
    
    return X

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    Yq = Y.copy()
    dwtent = dwtstep.copy()

    for (k, i), _ in np.ndenumerate(dwtstep):
        m = Y.shape[0] // (2 ** (i+1))
        if i + 1 == dwtstep.shape[1]:
            if k == 0:
                dwtent[k, i] = entropy(Yq[:m, :m])
                Yq[:m, :m] = quantise(Yq[:m, :m], dwtstep[0, i])
        else:
            match k:
                case 1:
                    dwtent[k, i] = entropy(Yq[m:2*m, :m])
                    Yq[m:2*m, :m] = quantise(Yq[m:2*m, :m], dwtstep[k, i])
                case 0:
                    dwtent[k, i] = entropy(Yq[:m, m:2*m])
                    Yq[:m, m:2*m] = quantise(Yq[:m, m:2*m], dwtstep[k, i])
                case 2:
                    dwtent[k, i] = entropy(Yq[m:2*m, m:2*m])
                    Yq[m:2*m, m:2*m] = quantise(Yq[m:2*m, m:2*m], dwtstep[k, i])

    return Yq, dwtent

def dwt_std(X, qnum, n, impulse_responce=None):
    if impulse_responce == None:
        impulse_responce = np.ones((3, n+1))
    if impulse_responce.shape != (3, n+1):
        raise ValueError(f"Impulse responce has shape {impulse_responce.shape}, but should be shape {(3, n+1)}")
    qnums = impulse_responce * qnum
    Y = nlevdwt(X, n)
    Yq = quantdwt(Y, qnums)[0]
    Z = nlevidwt(Yq, n)
    return np.std(X-Z)

@lru_cache
def calculate_impulse_responce(n):
    impulse_responce = np.zeros((3, n+1))
    impulse_responce.fill(np.nan)
    for (k, i), _ in np.ndenumerate(impulse_responce):
        Y = np.zeros((256, 256))
        m = Y.shape[0] // (2 ** (i+1))
        impulse_location = None
        if i + 1 == impulse_responce.shape[1]:
            if k == 0:
                impulse_location = np.s_[int(0.5*m), int(0.5*m)]
        else:
            match k:
                case 1:
                    impulse_location = np.s_[int(1.5*m), int(0.5*m)]
                case 0:
                    impulse_location = np.s_[int(0.5*m), int(1.5*m)]
                case 2:
                    impulse_location = np.s_[int(1.5*m), int(1.5*m)]
        if impulse_location:
            Y[impulse_location] = 100
            impulse_responce[k, i] = energy(nlevidwt(Y, n)) / 10000
            Y[impulse_location] = 0


    impulse_responce /= impulse_responce[0,0]
    impulse_responce = 1/np.sqrt(impulse_responce)