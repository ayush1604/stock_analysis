import get_data
import os
import pandas as pd

list_dir = 'stocks_list'
list_prefix = "ind_nifty"
list_suffix = "list.csv"
save_dir = 'ohlcv_data'
save_prefix = "ohlcv_"
save_suffix = ".csv"


if __name__ == "__main__":
    # for f in os.listdir(list_dir):
    #     if f.startswith(list_prefix) and f.endswith(list_suffix):
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         savefile = os.path.join(save_dir, save_prefix+f[9:-8]+save_suffix)
    #         get_data.get_from_list(os.path.join(list_dir, f), configfile='list_config.txt', savefile=savefile,
    #                                period="max", interval="1d", start=None, end=None, ticker_suffix='.NS')

    get_data.get_from_config(config = {
            'stocks' : '^NSEI',
            'period' : 'max',
            'interval' : '1d',
            'start' : None,
            'end' : None
        }, savefile='ohlcv_data/index_ohlcv_nse.csv')