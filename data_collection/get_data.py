import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import ast


def get_from_config(config=None, configfile=None, savefile=None):
    '''
    Download and return data as specified in config. Optionally save it.
    savefile : Name (shall include '.csv') of the save file.
    '''

    if config is None and configfile is not None:
        with open(configfile) as f:
            config = ast.literal_eval(f.read())

    if config['period'] is not None:
        df = yf.download(config['stocks'], period=config['period'], interval=config['interval'], threads=True)
    else:
        df = yf.download(config['stocks'], interval=config['interval'], start=config['start'], end=config['end'],
                         threads=True)

    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    # df.set_index('Date', inplace=True)
    if savefile is not None:
        df.to_csv(savefile)
        print('Saved '+savefile)
        with open('log.txt', 'a') as logfile:
            logfile.write("\n{} : {} saved with {} stocks for period {} to {}".format(
                datetime.now(), savefile, ', '.join(config['stocks']), df.index[0], df.index[-1]))

    return df


def get_from_list(listfile, configfile=None, savefile=None, period="max", interval="1d", start=None, end=None, ticker_suffix='.NS'):
    p = pd.read_csv(listfile)
    symbols = list(p['Symbol'].values + ticker_suffix)
    if configfile is not None:
        with open(configfile) as f:
            config = ast.literal_eval(f.read())
    else:
        config = {
            'period' : period,
            'interval' : interval,
            'start' : start,
            'end' : end
        }
    config['stocks'] = symbols
    get_from_config(config=config, savefile=savefile)


