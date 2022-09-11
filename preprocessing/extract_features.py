from fileinput import close
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from numpy.lib.stride_tricks import sliding_window_view
from preprocessing.wrangling import merge_in_one
from preprocessing.denoising import wavelet_denoising
from numpy import mean, absolute
import ta

def get_all_ta_features(df, name_open="Open", name_high="High", name_low="Low", name_close="Close", name_volume="Volume",
fillna=True):
    df = ta.utils.dropna(df)
    df = ta.add_all_ta_features(df, open=name_open, high=name_high, low=name_low, close=name_close, volume=name_volume, fillna=fillna)

    return df


def extract_ta_features(closing, high, low, volume, nd_ma=10, nd_momentum=10, nd_k=14, nd_d=3, nd_cci=20):
    'Depracted. Use get_all_ta_features instead.'
    window_ma = sliding_window_view(closing, window_shape=nd_ma)
    ma = pd.Series(
        np.pad(np.average(window_ma, axis=1), pad_width=(nd_ma - 1, 0), mode='constant', constant_values=np.nan))
    wma = pd.Series(np.pad(np.average(window_ma, weights=np.arange(1, nd_ma + 1), axis=1), pad_width=(nd_ma - 1, 0),
                           mode='constant', constant_values=np.nan))
    momentum = closing.diff(periods=nd_momentum)

    momentum_k = closing.diff(periods=nd_k)
    window_high = sliding_window_view(high, window_shape=nd_k)
    window_low = sliding_window_view(low, window_shape=nd_k)
    HH = pd.Series(
        np.pad(np.max(window_high, axis=1), pad_width=(nd_k - 1, 0), mode='constant', constant_values=np.nan))
    LL = pd.Series(np.pad(np.min(window_low, axis=1), pad_width=(nd_k - 1, 0), mode='constant', constant_values=np.nan))
    stochK = pd.Series(100 * ((closing - LL) / (HH - LL)))
    window_stochK = sliding_window_view(stochK, window_shape=nd_d)
    stochD = pd.Series(
        np.pad(np.average(window_stochK, axis=1), pad_width=(nd_d - 1, 0), mode='constant', constant_values=np.nan))

    up = pd.Series(np.zeros_like(closing.values))
    down = pd.Series(np.zeros_like(closing.values))
    change = closing.reset_index(drop=True).diff()
    up[change > 0] = change[change > 0]
    down[change < 0] = change[change < 0] * (-1)

    mau14 = up.ewm(alpha=1 / 14).mean()
    mad14 = down.ewm(alpha=1 / 14).mean()
    rsi = 100 - (100 / (1 + (mau14 / mad14)))

    ema12 = closing.ewm(span=12).mean()
    ema26 = closing.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    larryR = -100 * (HH - closing.values) / (HH - LL)

    mfm = (2 * closing - low - high) / (high - low)
    mfv = mfm * volume
    adl = [volume[0]]
    for i in range(1, len(mfm)):
        adl.append(adl[-1] + volume[i])
    adl = pd.Series(adl)

    typ_price = (high + low + closing) / 3
    typ_price.reset_index(drop=True, inplace=True)
    window_tp = sliding_window_view(typ_price, window_shape=nd_cci)
    ma_tp = pd.Series(
        np.pad(np.average(window_tp, axis=1), pad_width=(nd_cci - 1, 0), mode='constant', constant_values=np.nan))

    def mad(data, axis=None):
        return np.mean(np.absolute(data - np.mean(data, axis)), axis)

    a = []
    for day in window_tp:
        a.append(mad(day))
    a = np.array(a)
    ma_tp_d = pd.Series(np.pad(np.array(a), pad_width=(nd_cci - 1, 0), mode='constant', constant_values=np.nan))
    cci = (typ_price - ma_tp) / (0.015 * ma_tp_d)

    return ma, wma, momentum, stochK, stochD, rsi, macd, larryR, adl, cci

def get_wavelet_coeffs(x, len_window, axis=1, decomp_level=1):
    x = pd.Series(x)
    xdf = sliding_window_view(x, (len_window), writeable=True)
    x_swdf = pd.DataFrame.from_records(xdf)
    x_swdf_sm_coeff = x_swdf.apply(wavelet_denoising, axis=axis, decomp_level=decomp_level)
    return x_swdf_sm_coeff.apply(lambda row : np.array(merge_in_one(row[1])))