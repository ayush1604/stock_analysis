import pywt
import numpy as np


def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet='haar', decomp_level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per", level=decomp_level)
    sigma = (1/0.6745) * madev(coeff[-decomp_level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per'), coeff

