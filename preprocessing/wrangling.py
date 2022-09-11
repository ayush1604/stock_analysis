import pandas as pd
import numpy as np
import os
import datetime
from numpy.lib.stride_tricks import sliding_window_view


def to_indi_csv( ticker, opfile, start_date=None, end_date=None, ohlcvfile=None, df=None):
    assert ohlcvfile is not None or df is not None, "Either OHLCV file or DataFrame must be provided."
    if df is None:
        df = pd.read_csv(ohlcvfile)
    df = df.dropna(subset=[ticker])
    t = df[['Date', 'Ticker', ticker]].pivot_table(index='Date', columns='Ticker')
    t.columns = t.columns.droplevel()
    t.reset_index(inplace=True)
    t['Date'] = pd.to_datetime(t['Date'])

    if start_date is not None:
        t = t[t['Date'] >= datetime.datetime.fromisoformat(start_date)]

    if end_date is not None:
        t = t[t['Date'] <= datetime.datetime.fromisoformat(end_date)]
    
    t.to_csv(opfile)


def get_indi_df(ticker, ohlcvfile=None, start_date=None, end_date=None, df=None):
    assert ohlcvfile is not None or df is not None, "Either OHLCV file or DataFrame must be provided."
    if df is None:
        df = pd.read_csv(ohlcvfile)
    df = df.dropna(subset=[ticker])
    t = df[['Date', 'Ticker', ticker]].pivot_table(index='Date', columns='Ticker')
    t.columns = t.columns.droplevel()
    t.reset_index(inplace=True)
    t['Date'] = pd.to_datetime(t['Date'])
 
    if start_date is not None:
        t = t[t['Date'] >= datetime.datetime.fromisoformat(start_date)]

    if end_date is not None:
        t = t[t['Date'] <= datetime.datetime.fromisoformat(end_date)]
 
    return t


def get_labels(cls : pd.Series):
    move_dir = pd.Series(np.zeros(cls.shape), index=cls.index)
    cls_tmrw = cls.shift(-1)
    move_dir[cls_tmrw > cls] = 1
    move_dir[cls_tmrw <= cls] = -1
    move_dir[0] = np.nan
    move_dir[-1] = np.nan
    return move_dir[1:-1], cls_tmrw

def slide_and_flatten(X, window_len):
    Xsw = sliding_window_view(X, (window_len, X.shape[1]))
    Xsw = Xsw.squeeze()
    if len(Xsw.shape) == 2:
        return Xsw
    return Xsw.reshape(Xsw.shape[0], Xsw.shape[1]*Xsw.shape[2])
    
def merge_in_one(row):
    res = []
    for e in row:
        res.extend(e)

    return res