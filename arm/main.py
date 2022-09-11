import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from matplotlib import pyplot as plt
import arm_encoder

encode = True
ohlcv_dir = '../data_collection/ohlcv_data'
encoded_csv_path = None

if __name__ == "__main__":
    if encode:
        df = arm_encoder.arm_encoder(ohlcv_dir, savepath='encoded_all.csv', ohlcv_prefix='ohlcv_',
                                           ohlcv_suffix='.csv', verbose=True)
    else:
        df = pd.read_csv(encoded_csv_path)

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Date' in df.columns:
        df.drop('Date', axis=1, inplace=True)
    df.fillna(value=False, inplace=True)
    
    
    # Find and Save Frequent Itemsets. Don't run if already saved.
    frequent_itemsets_ap = apriori(df, min_support=0.2, use_colnames=True)
    frequent_itemsets_ap.to_csv('frequent_with_support_2perc.csv')
    
    # Load saved frequent itemsets and mine association rules.
    # Don't run if already mined.
    frequent_itemsets_ap = pd.read_csv('frequent_with_support_2perc.csv')
    rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.90)
    
    rules_ap.sort_values(by='confidence', ascending=False, inplace=True)
    rules_ap.to_csv('ar_with_conf_90perc.csv')