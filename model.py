#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import warnings
# warnings.filterwarnings("error")


# In[10]:

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as torchf
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_window_view
import pickle
import datetime
import math
import csv
from sklearn.model_selection import train_test_split
import itertools


# In[11]:


class NSEDataset(Dataset):
    def __init__(self, ohlcv_dir, target_ticker, target_ticker_file, len_window, len_corr_traceback, nP, nN, 
    keep_tickers=None, ohlcv_prefix='', ohlcv_sufix='', ohlcv_files=None, start_date=None, end_date=None,
    target_feat='c', keep_feat='ohlcva', normalize='min-max', normalize_target=False, corr_control=None, corr_threshold=None):
        '''
        corr_control : None - No control
                    'threshold' - Upper and lower threshold (corr_threshold must be specified.
                    'prop' - Weightage proportional to absolute correlation is given.

        corr_threshold : dict with two keys, p_threshold and n_threshold (not absolute). 
        '''

        feat_name_map = {
            'o' : 'Open', 
            'h' : 'High', 
            'l' : 'Low', 
            'c' : 'Close', 
            'v' : 'Volume',
            'a' : 'Adj Close'
        }

        self.len_window = len_window
        self.len_corr_traceback = len_corr_traceback
        self.nP, self.nN = nP, nN
        self.target_feat = target_feat
        self.keep_feat = keep_feat
        self.start_date, self.end_date = start_date, end_date
        if corr_control == 'threshold' and corr_threshold is None:
            raise ValueError("corr_threshold must be specified for corr_control = 'threshold.")
        self.corr_control = corr_control
        self.corr_threshold = corr_threshold

        if ohlcv_files is not None:
            ohlcv_files = set(ohlcv_files)
        
        if keep_tickers is not None:
            keep_tickers = set(keep_tickers)
    
        df = pd.read_csv(os.path.join(ohlcv_dir, target_ticker_file))
        df['Date'] = pd.to_datetime(df['Date'])

        if start_date is not None:
            start_mask =  df['Date'] >= datetime.datetime.fromisoformat(start_date)
            i_start = start_mask[start_mask].index.min()
        else:
            i_start = 0
        
        if end_date is not None:
            end_mask =  df['Date'] > datetime.datetime.fromisoformat(end_date)
            i_end = end_mask[end_mask].index.min()
        else:
            i_end = len(df)

        df.reset_index(drop=True, inplace=True)
        df.set_index(['Date', 'Ticker'], inplace=True)
        df = df.iloc[i_start:i_end]
        
        self.mainstream_df = df.loc[:, target_ticker]
        self.df = df.drop(target_ticker, axis=1)
        
        if ohlcv_files is not None and target_ticker_file not in ohlcv_files:
            self.df = pd.DataFrame(columns=df.columns)
            
        for f in os.listdir(ohlcv_dir):
            if f.startswith(ohlcv_prefix) and f.endswith(ohlcv_sufix) and (ohlcv_files is None or f in ohlcv_files):
                if f == target_ticker_file:
                    continue
                else:
                    temp_df = pd.read_csv(os.path.join(ohlcv_dir, f))
                    temp_df.reset_index(drop=True, inplace=True)
                    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                    self.df = pd.merge(self.df.reset_index(), temp_df, on=['Date', 'Ticker'],
                    how='inner', suffixes=('', '_y')).set_index(['Date', 'Ticker'])
                    self.df.drop(self.df.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
                    
        if keep_tickers is not None:
            for c in self.df:
                if c not in keep_tickers:
                    self.df.drop(c, axis=1, inplace=True)

        
        self.df = self.df.pivot_table(index='Date', columns='Ticker')
        self.df.columns = self.df.columns.map('_'.join)
        if isinstance(self.mainstream_df, pd.Series):
            self.mainstream_df = pd.DataFrame(self.mainstream_df)
        
        self.mainstream_df = self.mainstream_df.pivot_table(index='Date', columns='Ticker')
        self.mainstream_df.columns = self.mainstream_df.columns.map('_'.join)

        target_feat_name = "{}_{}".format(target_ticker, feat_name_map[target_feat])
        
        self.unshifted_target = self.mainstream_df.loc[:, target_feat_name]
        self.target = self.unshifted_target.shift(periods=-1).iloc[:-1]

        # To account for absence of target for last row.
        self.df = self.df.iloc[:-1, :]  
        self.mainstream_df = self.mainstream_df.iloc[:-1, :]

        drop_features = set(feat_name_map.keys()).difference({feat for feat in keep_feat})
        for feat in drop_features:
            self.df.drop(self.df.filter(regex='_{}$'.format(feat_name_map[feat])).columns.tolist(), axis=1, inplace=True)
            self.mainstream_df.drop(self.mainstream_df.filter(regex='_{}$'.format(feat_name_map[feat])).columns.tolist(), axis=1, inplace=True)


        # For i_end, data of [i_end - (len_corr_traceback) : i_end] (py notation)
        # is needed to calculate correlation, basis on which data of 
        # [i_end - (len_window) : i_end] (py notation) must be sent.
        # i_end is excluded, so last i_end should be len(self.df)
        
        self.min_chosen_p = float('inf')
        self.max_chosen_n = float('-inf')
        self.swdf = []
        self.swcorr = []
        for i_end in range(len_corr_traceback, len(self.df)+1):
            newrow, corrrow = self.get_high_corr(self.unshifted_target.iloc[i_end-len_corr_traceback:i_end], 
            self.df.iloc[i_end-len_corr_traceback:i_end, :], len_window, nP, nN)
            self.swdf.append(newrow)
            self.swcorr.append(corrrow)
            
        # self.swdf = np.array(self.swdf).reshape(len(self.swdf), self.len_window, -1)
        self.swdf = np.array(self.swdf)
        self.swcorr = np.array(self.swcorr)
        self.swdf = pd.DataFrame(self.swdf).fillna(method='ffill').to_numpy()
         
        # if earlier self.df.shape was (6(n+1), c), it should now be
        # (n, c), mainstream_df.shape and index_data_df should be (n, 1) and swdf.shape
        # should be (n-lct+1, lw*(nP+nN)).

        # For index 0, 
        # swdf[0], mainstream_df[lct-lw : lct] flattend (py notation)
        # index_data_df[lct-lw : lct] flattened (py notation) should be accessed.

        # For index i < len(swdf), 
        # swdf[i], mainstream_df[lct-lw + i: lct + i] flattend (py notation)
        # index_data_df[lct-lw + i : lct + i] flattened (py notation) should be accessed.
        # Correct. Continue from here. 

        # The index data used is for a single index.
        self.index_data_df = pd.read_csv("data_collection/NIFTY 50.csv")
        self.index_data_df['Date'] = pd.to_datetime(self.index_data_df['Date'])
        self.index_data_df.rename(columns={'SharesTraded' : 'Volume'}, inplace=True)
        self.index_data_df.drop("Unnamed: 0", axis=1, inplace=True)

        if start_date is not None:
            start_mask =  self.index_data_df['Date'] >= datetime.datetime.fromisoformat(start_date)
            i_start = start_mask[start_mask].index.min()
        else:
            i_start = 0

        if end_date is not None:
            end_mask =  self.index_data_df['Date'] > datetime.datetime.fromisoformat(end_date)
            i_end = end_mask[end_mask].index.min()
        else:
            i_end = len(self.index_data_df)

        self.index_data_df.reset_index(drop=True, inplace=True)
        self.index_data_df.set_index(['Date'], inplace=True)
        self.index_data_df = self.index_data_df.iloc[i_start:i_end]

        self.index_data_df = pd.DataFrame(self.index_data_df.loc[:, 'Close'])
        self.index_data_df = self.index_data_df.reset_index().merge(
    self.df.reset_index()['Date'], how='inner', on='Date').set_index('Date')
        
        # Min-max normalizing.
        if normalize == 'min-max':
            self.index_data_df=(self.index_data_df-self.index_data_df.min())/(self.index_data_df.max()-self.index_data_df.min())
            self.swdf = (self.swdf-self.swdf.min())/(self.swdf.max()-self.swdf.min())
            self.mainstream_df = (self.mainstream_df-self.mainstream_df.min())/(self.mainstream_df.max()-self.mainstream_df.min())
            self.df = (self.df-self.df.min())/(self.df.max()-self.df.min())
            if normalize_target:
                self.target = (self.target - self.target.min())/(self.target.max() - self.target.min())
                self.unshifted_target = (self.unshifted_target - self.unshifted_target.min())/(self.unshifted_target.max() - self.unshifted_target.min())
        

        print("Dataset created for {}".format(target_ticker))

    def get_high_corr(self, target: pd.Series, candidates: pd.DataFrame, len_window, nP, nN):
        corr = candidates.corrwith(target)
        p_best = corr.nlargest(nP)
        n_best = corr.nsmallest(nN)
        self.min_chosen_p = min(self.min_chosen_p, min(p_best))
        self.max_chosen_n = max(self.max_chosen_n, max(n_best))
        assert not p_best.isna().any()
        assert not n_best.isna().any()        
        newrow = candidates.iloc[-len_window:, candidates.columns.get_indexer(p_best.index)].melt()['value'].tolist()
        newrow.extend(candidates.iloc[-len_window:, candidates.columns.get_indexer(n_best.index)].melt()['value'].tolist())
        corrrow = p_best.tolist() + n_best.tolist()
        return newrow, corrrow


    def __len__(self):
        return len(self.swdf)

    def __getitem__(self, idx):
        # For index i < len(swdf), 
        # swdf[i], mainstream_df[lct-lw + i: lct + i] flattend (py notation)
        # index_data_df[lct-lw + i : lct + i] flattened (py notation) should be accessed.
        
        return (self.swdf[idx, :].reshape(self.len_window, -1), 
        self.swcorr[idx, :],
        self.mainstream_df.iloc[self.len_corr_traceback-self.len_window+idx : self.len_corr_traceback+idx].to_numpy(), 
        self.index_data_df.iloc[self.len_corr_traceback-self.len_window+idx : self.len_corr_traceback+idx].to_numpy(),
        self.target[self.len_corr_traceback+idx-1]
        )


def save_NSEDataset(dataset, opfile):
    with open(opfile, 'wb') as f:
        pickle.dump(dataset, f)

def load_NSEDataset(ipfile):
    PICKLE_PROTOCOL = 4
    with open(ipfile, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset 


# In[12]:


class MiLSTM(nn.Module):
   def __init__(self, input_sz: int, hidden_sz: int):
       super().__init__()
       self.input_size = input_sz
       self.hidden_size = hidden_sz
       self.p_size = input_sz 
       self.n_size = input_sz 
       self.index_size = input_sz
       p_sz, n_sz, index_sz = self.p_size, self.n_size, self.index_size

       #f_t
       self.Wfh = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wfy = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
       self.bf = nn.Parameter(torch.Tensor(hidden_sz))
       
       #o_t
       self.Woh = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Woy = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
       self.bo = nn.Parameter(torch.Tensor(hidden_sz))
       
       #c_t
       self.Wch = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wcy = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
       self.bc = nn.Parameter(torch.Tensor(hidden_sz))
       
       #c_pt
       self.Wcph = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wcpp = nn.Parameter(torch.Tensor(p_sz, hidden_sz))
       self.bcp = nn.Parameter(torch.Tensor(hidden_sz))

       #c_nt
       self.Wcnh = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wcnn = nn.Parameter(torch.Tensor(n_sz, hidden_sz))
       self.bcn = nn.Parameter(torch.Tensor(hidden_sz))
       
       #c_it
       self.Wcih = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wcii = nn.Parameter(torch.Tensor(index_sz, hidden_sz))
       self.bci = nn.Parameter(torch.Tensor(hidden_sz))

       #i_t
       self.Wih = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wiy = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
       self.bi = nn.Parameter(torch.Tensor(hidden_sz))
       
       #i_pt
       self.Wiph = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wipy = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
       self.bip = nn.Parameter(torch.Tensor(hidden_sz))

       #c_nt
       self.Winh = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Winy = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
       self.bin = nn.Parameter(torch.Tensor(hidden_sz))
       
       #c_it
       self.Wiih = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.Wiiy = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
       self.bii = nn.Parameter(torch.Tensor(hidden_sz))

       #attn
       self.alpha_t = nn.Parameter(torch.Tensor(1))
       self.alpha_pt = nn.Parameter(torch.Tensor(1))
       self.alpha_nt = nn.Parameter(torch.Tensor(1))
       self.alpha_it = nn.Parameter(torch.Tensor(1))
       self.Wattn = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
       self.ba = nn.Parameter(torch.Tensor(1))
       self.bap = nn.Parameter(torch.Tensor(1))
       self.ban = nn.Parameter(torch.Tensor(1))
       self.bai = nn.Parameter(torch.Tensor(1))

       self.init_weights()

   
   def init_weights(self):
       stdv = 1.0 / math.sqrt(self.hidden_size)
       for weight in self.parameters():
           weight.data.uniform_(-stdv, stdv)

       #c_pt
       nn.init.zeros_(self.Wcph)
       nn.init.zeros_(self.Wcpp)
       nn.init.zeros_(self.bcp)
       
       #c_nt
       nn.init.zeros_(self.Wcnh)
       nn.init.zeros_(self.Wcnn)
       nn.init.zeros_(self.bcn)
       
       #c_it
       nn.init.zeros_(self.Wcih)
       nn.init.zeros_(self.Wcii)
       nn.init.zeros_(self.bci)
       

   def forward(self, y_tilde, p_tilde, n_tilde, index_tilde, init_stats=None):
       batch_size, win_len, _ = y_tilde.shape
       hidden_seqs = []
       cell_states = []

       if init_stats is None:
           h_t, cell_t = (torch.zeros(batch_size, self.hidden_size).to(y_tilde.device), 
                       torch.zeros(batch_size, self.hidden_size).to(y_tilde.device))
       else:
           h_t, cell_t = init_states 

       
       for t in range(win_len):
           y_t = y_tilde[:, t, :]
           p_t = p_tilde[:, t, :]
           n_t = n_tilde[:, t, :]
           index_t = index_tilde[:, t, :]

           f_t = torch.sigmoid(y_t @ self.Wfy + h_t @ self.Wfh + self.bf)
           o_t = torch.sigmoid(y_t @ self.Woy + h_t @ self.Woh + self.bo)
           c_t = torch.tanh(y_t @ self.Wcy + h_t @ self.Wch + self.bc)
           c_pt = torch.tanh(p_t @ self.Wcpp + h_t @ self.Wcph + self.bcp)
           c_nt = torch.tanh(n_t @ self.Wcnn + h_t @ self.Wcnh + self.bcn)
           c_it = torch.tanh(index_t @ self.Wcii + h_t @ self.Wcph + self.bci)

           i_t = torch.sigmoid(y_t @ self.Wiy + h_t @ self.Wih + self.bi)
           i_pt = torch.sigmoid(y_t @ self.Wipy + h_t @ self.Wiph + self.bip)
           i_nt = torch.sigmoid(y_t @ self.Winy + h_t @ self.Winh + self.bin)
           i_it  = torch.sigmoid(y_t @ self.Wiiy + h_t @ self.Wiih + self.bii)

           l_t = torch.mul(c_t, i_t)
           l_pt = torch.mul(c_pt, i_pt)
           l_nt = torch.mul(c_nt, i_nt)
           l_it = torch.mul(c_it, i_it)
          
           u_t = torch.mul(l_t @ self.Wattn, cell_t).sum(dim=1)
           u_pt = torch.mul(l_pt @ self.Wattn, cell_t).sum(dim=1)
           u_nt = torch.mul(l_nt @ self.Wattn, cell_t).sum(dim=1)
           u_it = torch.mul(l_it @ self.Wattn, cell_t).sum(dim=1)

           alphas = torch.stack((u_t, u_pt, u_nt, u_it), dim=1)
           softmax = nn.Softmax(dim=1)
           probs = softmax(alphas)
           alpha_t, alpha_pt, alpha_nt, alpha_it = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
           
           L_t = self.alpha_t*l_t + self.alpha_pt*l_pt + self.alpha_nt*l_nt + self.alpha_it*l_it

           cell_t = torch.mul(cell_t, f_t) + L_t
           h_t = torch.mul(torch.tanh(cell_t), o_t)
           
           hidden_seqs.append(h_t)
           cell_states.append(cell_t)

       hidden_seqs = torch.stack(hidden_seqs)
       hidden_seqs = hidden_seqs.transpose(0, 1).contiguous()
       return hidden_seqs, (h_t, cell_t)


# In[13]:


class NeuralNetwork(nn.Module):
    def __init__(self, keep_features, hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3, nP, nN, lstm1_layers=1, lstm1_dropout=0):
        super(NeuralNetwork, self).__init__()
        self.hidden_sz1 = hidden_sz1
        self.hidden_sz2 = hidden_sz2
        self.hidden_sz_lin1, self.hidden_sz_lin2, self.hidden_sz_lin3 = hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3
        self.lstm = nn.LSTM(input_size = len(keep_features),
        hidden_size = hidden_sz1,
        num_layers = lstm1_layers,
        batch_first = True,
        dropout = lstm1_dropout
        )

        self.milstm = MiLSTM(hidden_sz1, hidden_sz2)
        self.self_attn_linear = nn.Linear(in_features=hidden_sz2, out_features=hidden_sz2, bias=True)
        self.self_attn_v = nn.Parameter(torch.rand(hidden_sz2))

        self.swdf_norm = nn.BatchNorm1d(nP+nN)
        self.mainstream_norm = nn.BatchNorm1d(len(keep_features))
        self.index_norm = nn.BatchNorm1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_sz2, hidden_sz_lin1),
            nn.LeakyReLU(),
            nn.Linear(hidden_sz_lin1, hidden_sz_lin2),
            nn.LeakyReLU(),
            nn.Linear(hidden_sz_lin2, hidden_sz_lin3),
            nn.LeakyReLU()
        )
        

    def forward(self, swdf, corr, mainstream, index, nP, nN, corr_control=None, corr_threshold=None):
        
        # Normalizing using nn.BatchNorm1d.
        # swdf = self.swdf_norm(swdf.permute(0, 2, 1)).permute(0, 2, 1)
        # mainstream = self.mainstream_norm(mainstream.permute(0, 2, 1)).permute(0, 2, 1)
        # index = self.index_norm(index.permute(0, 2, 1)).permute(0, 2, 1)
        y_tilde, _ = self.lstm(mainstream)
        pcorr, ncorr = corr[:, :nP], corr[:, nP:nP+nN]
        
        Xps = []
        for i_feature in range(nP):
            Xpi, _ = self.lstm(swdf[:, :, i_feature].unsqueeze(2))
            Xps.append(Xpi)
        Xps = torch.stack(Xps)

        Xns = []
        for i_feature in range(nP, nP+nN):
            Xni, _ = self.lstm(swdf[:, :,i_feature].unsqueeze(2))
            Xns.append(Xni)
        Xns = torch.stack(Xns)

        if corr_control == 'threshold':
            pth, nth = corr_threshold['p_threshold'], corr_threshold['n_threshold']
            pcorr[pcorr >= pth] = 1
            pcorr[pcorr < pth] = 0
            ncorr[ncorr <= nth] = 1
            ncorr[ncorr > nth] = 0
        elif corr_control == 'prop':
            pcorr, ncorr = torchf.normalize(pcorr, p=1, dim=1), torchf.normalize(ncorr, p=1, dim=1)

        if corr_control is not None:
            Xps = Xps.permute(2, 3, 1, 0)
            Xps = Xps * torch.abs(pcorr)
            Xps = Xps.permute(3, 2, 0, 1)

            Xns = Xns.permute(2, 3, 1, 0)
            Xns = Xns * torch.abs(ncorr)
            Xns = Xns.permute(3, 2, 0, 1)

        p_tilde = Xps.mean(axis=0)
        n_tilde = Xns.mean(axis=0)
        
        index_tilde, _ = self.lstm(index)
        # print(y_tilde.shape, p_tilde.shape, n_tilde.shape, index_tilde.shape)
        
        y_tilde_prime, _ = self.milstm(y_tilde, p_tilde, n_tilde, index_tilde)

        js_ = self.self_attn_linear(y_tilde_prime)
        js = torch.tanh(js_).matmul(self.self_attn_v)
        betas = torch.softmax(js, dim=1)
        y_ = torch.matmul(betas.unsqueeze(1), y_tilde_prime).squeeze()
        y_hat = self.regressor(y_).squeeze()
        return y_hat


# In[14]:


def cross_validate(dataloader, model, loss_fn, optimizer, test_size, corr_control=None, corr_threshold=None):
    train_mse_record = []
    train_mape_record = []
    train_weights = []

    test_mse_record = []
    test_mape_record = []
    test_weights = []


    for i, sample in enumerate(dataloader):
        swdf, corr, mainstream, index, y_act = sample
        swdf, corr, mainstream, index, y_act = swdf.float(), corr.float(), mainstream.float(), index.float(), y_act.float()
        swdf_tr, swdf_ts, corr_tr, corr_ts, mainstream_tr, mainstream_ts, index_tr, index_ts, y_act_tr, y_act_ts = train_test_split(swdf, corr, mainstream, index, y_act, test_size=test_size)

        pred = model(swdf_tr, corr_tr, mainstream_tr, index_tr, dataloader.dataset.nP, dataloader.dataset.nN)
        loss = loss_fn(pred, y_act_tr)
        mape = torch.mean((torch.abs((y_act_tr - pred) / y_act_tr)) * 100)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_mse_record.append(loss.item())
        train_mape_record.append(mape.item())
        train_weights.append(len(swdf_tr))

        print('i ', i)
        # Evaluation
        model.eval()
        with torch.no_grad():
            pred = model(swdf_ts, corr_ts, mainstream_ts, index_ts, dataloader.dataset.nP, dataloader.dataset.nN, corr_control, corr_threshold)
            loss = loss_fn(pred, y_act_ts)
            mape = torch.mean((torch.abs((y_act_ts - pred) / y_act_ts)) * 100)
            test_mse_record.append(loss.item())
            test_mape_record.append(mape.item())
            test_weights.append(len(swdf_ts))
        
    return np.average(train_mse_record, weights=train_weights), np.average(train_mape_record, weights=train_weights), np.average(test_mse_record, weights=test_weights), np.average(test_mape_record, weights=test_weights)


def eval_cv(dataset_path, batch_size, keep_features, hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3,
                  learning_rate, epochs, clip_gradients=False, corr_control=None, corr_threshold=None, save_comment=''):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    dataset = load_NSEDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    nP, nN = dataloader.dataset.nP, dataloader.dataset.nN 

    model = NeuralNetwork(keep_features=keep_features, hidden_sz1=hidden_sz1, hidden_sz2=hidden_sz2, 
    hidden_sz_lin1=hidden_sz_lin1, hidden_sz_lin2=hidden_sz_lin2, hidden_sz_lin3=hidden_sz_lin3, nP=nP, nN=nN).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    if clip_gradients:
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -50, 50))

    mse_tr_epochs = []
    mape_tr_epochs = []
    mse_ts_epochs = []
    mape_ts_epochs = []

    for i_epoch in range(epochs):
        mse_tr, mape_tr, mse_ts, mape_ts = cross_validate(dataloader, model, loss_fn, optimizer, test_size=0.2, 
        corr_control=corr_control, corr_threshold=corr_threshold)
        mse_tr_epochs.append(mse_tr)
        mape_tr_epochs.append(mape_tr)
        mse_ts_epochs.append(mse_ts)
        mape_ts_epochs.append(mape_ts)

        
    print("Cross Validation Completed for {}".format(dataset_path))
    
    min_mse_tr = min(mse_tr_epochs)
    min_mse_ts = min(mse_ts_epochs)
    min_mape_tr = min(mape_tr_epochs)
    min_mape_ts = min(mape_ts_epochs)

    last_mse_tr = mse_tr_epochs[-1]
    last_mse_ts = mse_ts_epochs[-1]
    last_mape_tr = mape_tr_epochs[-1]
    last_mape_ts = mape_ts_epochs[-1]

    with open('results_cv_corr_control_itc.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([datetime.datetime.now(), last_mse_tr, last_mse_ts, last_mape_tr, last_mape_ts, min_mse_tr, min_mse_ts, min_mape_ts, min_mape_ts, nP, nN, epochs, learning_rate, save_comment, dataset_path])
        
    print("Results file updated.")
 

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    mse_record = []
    mape_record = []
    model.train()
    for i, sample in enumerate(dataloader):
        swdf, mainstream, index, y_act = sample
        swdf, mainstream, index, y_act = swdf.float(), mainstream.float(), index.float(), y_act.float()
        pred = model(swdf, mainstream, index, dataloader.dataset.nP, dataloader.dataset.nN)
        loss = loss_fn(pred, y_act)
        mape = torch.mean((torch.abs((y_act - pred) / y_act)) * 100)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse_record.append(loss)
        mape_record.append(mape)
        
        if i % 1 == 0:
            loss, current = loss.item(), i * len(swdf)
            print(f"MSE loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return mse_record, mape_record

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    mse_record = []
    mape_record = []
    model.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            swdf, mainstream, index, y_act = sample
            swdf, mainstream, index, y_act = swdf.float(), mainstream.float(), index.float(), y_act.float()
            pred = model(swdf, mainstream, index, dataloader.dataset.nP, dataloader.dataset.nN)
            loss = loss_fn(pred, y_act)
            mape = torch.mean((torch.abs((y_act - pred) / y_act)) * 100)
            mse_record.append(loss)
            mape_record.append(mape)
    
    return mse_record, mape_record
            
# In[15]:

def train_and_test(train_dataset_path, test_datset_path, batch_size, keep_features, 
                  hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3,
                  learning_rate, epochs, clip_gradients=False):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    train_dataset = load_NSEDataset(train_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    nP, nN = train_dataloader.dataset.nP, train_dataloader.dataset.nN 

    model = NeuralNetwork(keep_features=keep_features, hidden_sz1=hidden_sz1, hidden_sz2=hidden_sz2, 
    hidden_sz_lin1=hidden_sz_lin1, hidden_sz_lin2=hidden_sz_lin2, hidden_sz_lin3=hidden_sz_lin3, nP=nP, nN=nN).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    if clip_gradients:
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -50, 50))

    mse_record = []
    mape_record = []
    mse_epochs = []
    mape_epochs = []
    for i_epoch in range(epochs):
        mse_, mape_ = train(train_dataloader, model, loss_fn, optimizer)
        mse_record.extend(mse_)
        mse_epochs.append(mse_[-1])
        mape_record.extend(mape_)
        mape_epochs.append(mape_[-1])
        print("Epoch {} completed.".format(i_epoch+1))
        
    print("Training completed for {}".format(train_dataset_path))
    
    min_mse_all = min(mse_record)
    avg_mse_all = torch.mean(torch.stack(mse_record))
    min_mape_all = min(mape_record)
    avg_mape_all = torch.mean(torch.stack(mape_record))
    
    test_dataset = load_NSEDataset(test_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    mse_test, mape_test = test_loop(test_dataloader, model, loss_fn)
    print("Testing completed for {}".format(test_dataset_path))
    
    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([datetime.datetime.now(), min_mse_all.item(), avg_mse_all.item(), min_mape_all.item(), avg_mape_all.item(), mse_test[0].item(), mape_test[0].item(), nP, nN, epochs, learning_rate, train_dataset.min_chosen_p, train_dataset.max_chosen_n, test_dataset.min_chosen_p, test_dataset.max_chosen_n, train_dataset_path, test_dataset_path])
        
    print("Results file updated.")
    return mse_test, mape_test
    

# In[16]:
list_nP = [50, 100, 150]
list_normalize = ['min-max']
list_corr_threshold = [
{'p_threshold' : 0.75,'n_threshold' : -0.75},
{'p_threshold' : 0.2,'n_threshold' : -0.2},
{'p_threshold' : 0,'n_threshold' : -0}]


for nP, normalize, corr_threshold in itertools.product(list_nP, list_normalize, list_corr_threshold):
    nN = nP
    # nP = 10
    # nN = 10
    ohlcv_dir = 'data_collection/ohlcv_data'
    target_ticker = 'ITC.NS'
    target_ticker_file = 'ohlcv_fmcg.csv'
    len_window = 10
    len_corr_traceback = 20
    keep_feat = 'o'
    # train_start_date = '2017-01-01'
    # train_end_date = None
    
    start_date = '2017-01-01'
    end_date = None 
    # For predictions to start from day i (2021-06-01), 
    # i should be (len_corr_traceback)th day in the test_dataloader.
    # test_start_date = '2021-05-13'
    # test_end_date = None
    # normalize = 'min-max'
    # normalize = None
    normalize_target = False
    corr_control = 'threshold'
    # corr_threshold = {'p_threshold' : 0.9,'n_threshold' : -0.9}

    dataset = NSEDataset(ohlcv_dir=ohlcv_dir, 
                         target_ticker=target_ticker, 
                         target_ticker_file=target_ticker_file, 
                         len_window=len_window,
                         len_corr_traceback=len_corr_traceback, 
                         nP=nP,
                         nN=nN, 
                         keep_feat=keep_feat, 
                         start_date=start_date,
                         end_date=end_date,
                         normalize=normalize,
                         normalize_target=False)

    # test_dataset = NSEDataset(ohlcv_dir=ohlcv_dir, 
    #                      target_ticker=target_ticker, 
    #                      target_ticker_file=target_ticker_file, 
    #                      len_window=len_window,
    #                      len_corr_traceback=len_corr_traceback, 
    #                      nP=nP,
    #                      nN=nN, 
    #                      keep_feat=keep_feat, 
    #                      start_date=test_start_date,
    #                      end_date=test_end_date,
    #                      normalize=normalize)

    end_date_str = end_date[:7] if end_date is not None else 'full'
    if corr_control is None:
        corr_control_str = 'NoCorrControl'
    elif corr_control == 'prop':
        corr_control_str = 'PropCorrControl'
    else:
        corr_control_str = 'Th{}_{}'.format(corr_threshold['p_threshold'], corr_threshold['n_threshold'])
    dataset_path = 'data_collection/pickled_datasets/{}_Normalized_targetNorm{}_{}_{}_{}_{}_w{}_t{}_p{}_n{}_{}.pkl'.format(
    normalize, normalize_target, corr_control_str, target_ticker[:-3], start_date[:7], end_date_str, len_window, len_corr_traceback, nP, nN, keep_feat)

    save_NSEDataset(dataset, dataset_path)

    batch_size = 256
    keep_features = 'o'
    hidden_sz1 = 128
    hidden_sz2 = 64
    hidden_sz_lin1 = 64
    hidden_sz_lin2 = 32
    hidden_sz_lin3 = 1
    learning_rate = 0.0005
    epochs = 30
    save_comment = "hs1{}_hs2{}_hslin1{}_hslin2{}_hslin3{}_lr{}_epochs{}".format(
        hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3, learning_rate, epochs
    )

    eval_cv(dataset_path, batch_size, keep_features, hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3,
    learning_rate, epochs, clip_gradients=False, corr_control=corr_control,
                         corr_threshold=corr_threshold, save_comment=save_comment)

    # train_and_test(train_dataset_path, test_dataset_path, batch_size, keep_features, 
    #                   hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3,
    #                   learning_rate, epochs, clip_gradients=False)


# In[ ]:


# epochs = 1
# mse_test, mape_test = train_and_test(train_dataset_path, test_dataset_path, batch_size, keep_features, 
#                   hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3,
#                   learning_rate, epochs, clip_gradients=False);

# eval_cv(dataset_path, batch_size, keep_features, hidden_sz1, hidden_sz2, hidden_sz_lin1, hidden_sz_lin2, hidden_sz_lin3,
                #   learning_rate, epochs, clip_gradients=False):torch.stack(Xns)