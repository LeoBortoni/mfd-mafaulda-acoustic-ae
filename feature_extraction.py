import numpy as np
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import rfft, rfftfreq
from sklearn.neighbors import KernelDensity


def rms_value(signal: np.ndarray):
    """
    root mean square
    """
    rms = np.sqrt(np.mean(signal**2))
    return rms


def sra_value(signal: np.ndarray):
    """
    square root of the amplitude
    """
    sra = (np.mean(np.sqrt(np.absolute(signal)))) ** 2
    return sra


def kv_value(signal: np.ndarray):
    """
    kurtosis value
    """
    kv = kurtosis(signal)
    return kv


def sv_value(signal: np.ndarray):
    """
    skewness value
    """
    sv = skew(signal)
    return sv


def ppv_value(signal: np.ndarray):
    """
    peak to peak value
    """
    return np.max(signal) - np.min(signal)


def cf_value(signal: np.ndarray):
    """
    crest factor
    """
    cf = np.max(np.absolute(signal)) / rms_value(signal)
    return cf


def if_value(signal: np.ndarray):
    """
    impulse value
    """
    _if = np.max(np.absolute(signal)) / np.mean(np.absolute(signal))
    return _if


def mf_value(signal: np.ndarray):
    """
    margin factor
    """
    mf = np.max(np.absolute(signal)) / sra_value(signal)
    return mf


def sf_value(signal: np.ndarray):
    """
    shape factor
    """
    sf = rms_value(signal) / np.mean(np.absolute(signal))
    return sf


def kf_value(signal: np.ndarray):
    """
    kurtosis factor
    """
    return kv_value(signal) / (rms_value(signal) ** 4)


def fc_value(signal: np.ndarray):
    """
    Frequency center
    """
    fft_normalized = 2 * np.abs(rfft(signal)) / signal.size
    return np.mean(fft_normalized)


def rmsf_value(signal: np.ndarray):
    """
    Root mean square frequency
    """
    fft_normalized = 2 * np.abs(rfft(signal)) / signal.size
    return np.sqrt(np.mean(fft_normalized**2))


def rvf_value(signal: np.ndarray):
    """
    Root variance frequency
    """
    fft_normalized = 2 * np.abs(rfft(signal)) / signal.size
    return np.sqrt(np.mean((fft_normalized - fc_value(signal)) ** 2))
