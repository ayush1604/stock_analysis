import pandas as pd
import numpy as np
import os

def arm_ud_encoder(series):
    diff = series.diff()
    u, d = diff > 0, diff < 0
    return u, d


def arm_encoder(ohlcv_dir, savepath=None, ohlcv_prefix='', ohlcv_suffix='', verbose=False):
    encoded_df = pd.DataFrame(columns=['Date', 'Ticker'])
    for file in os.listdir(ohlcv_dir):
        if file.startswith(ohlcv_prefix) and file.endswith(ohlcv_suffix):
            df = pd.read_csv(os.path.join(ohlcv_dir, file), index_col='Date')
            df = df[df['Ticker'] == 'Adj Close']

            for c in df.columns:
                if c == 'Date' or c == 'Ticker':
                    continue
                u, d = arm_ud_encoder(df[c])
                uname, dname = c + '_up', c + '_down'
                u.rename(uname, inplace=True)
                d.rename(dname, inplace=True)
                encoded_df = encoded_df.merge(u, on='Date', how='outer')
                encoded_df = encoded_df.merge(d, on='Date', how='outer')

                if verbose:
                    print("Encoded {}".format(c))

    if savepath is not None:
        encoded_df.to_csv(savepath)
        if verbose:
            print("Saved {}.".format(savepath))

    return encoded_df
