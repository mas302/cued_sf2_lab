import numpy as np

from cued_sf2_lab.laplacian_pyramid import rowint, rowdec, quantise

def downscale(X, h):
    return rowdec(rowdec(X , h).T, h).T

def upscale(X, h):
    return rowint(rowint(X, 2*h).T, 2*h).T

def lp_decode(*args):
    if len(args) == 3:
        return [args[0] + upscale(args[1], args[2])]
    else:
        subpydec = lp_decode(*args[1::])
        Z = args[0] + upscale(subpydec[-1], args[-1])
        return *subpydec, Z

def lp_encode(X, n, h):
    if n == 0:
        return [X]
    Xn = downscale(X, h)
    return X - upscale(Xn, h), *lp_encode(Xn, n-1, h)

def dct_std(X, qnum, n, h):
    Ys = lp_encode(X, n, h)
    Yq = [quantise(Y, qnum) for Y in Ys]
    Z = lp_decode(*Yq, h)
    return np.std(X-Z)