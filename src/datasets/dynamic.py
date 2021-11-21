import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Dynamic(Dataset):
    # load the dataset
    def __init__(self, gages_dir, X_dir, y_dir, beg_year, beg_month,
                 seq_length, input_size_dyn, output_size, eco=None):
        # load baseflow csv to a dataframe
        flow = pd.DataFrame()
        gages = pd.read_excel(gages_dir, usecols=['STAID', 'ECO'], dtype=str)
        if eco is not None:
            gages = gages[gages['ECO'] == eco].reset_index(drop=True)
        gages['num_months'] = -1
        for i, basin in gages.iterrows():
            df = pd.read_csv(os.path.join(y_dir, basin['STAID'] + '.csv'))
            flow_beg = np.searchsorted((df['Y'] == beg_year + (beg_month + seq_length - 2) // 12) &
                                       (df['M'] >= (beg_month + seq_length - 2) % 12 + 1) |
                                       (df['Y'] >= beg_year + (beg_month + seq_length - 2) // 12 + 1), True)
            if flow_beg < df.shape[0]:
                flow = pd.concat([flow, df.iloc[flow_beg:]], axis=0, ignore_index=True)
                gages.loc[i, 'num_months'] = df.iloc[flow_beg:].shape[0]

        # assign time deltas and basin indices to the baseflow dataframe
        flow['delta'] = (flow['Y'] - beg_year) * 12 + flow['M'] - beg_month
        gages = gages[gages['num_months'] != -1]
        gages['t'] = np.cumsum(gages['num_months'])
        gages['s'] = gages['t'].shift(1, fill_value=0)
        flow['basin'] = np.repeat(np.arange(0, gages.shape[0], dtype=int), gages['t'] - gages['s'])

        # construct X
        X = np.zeros((flow.shape[0], seq_length, input_size_dyn))
        for i, basin in gages.iterrows():
            df = pd.read_csv(os.path.join(X_dir, basin['STAID'] + '.csv'))
            basin_idx = flow['basin'] == i
            delta = flow['delta'][basin_idx]
            idx_sub = np.tile(np.arange(0, seq_length)[::-1], delta.shape[0])
            idx_rep = np.repeat(delta, seq_length)
            X[basin_idx] = np.array(df.iloc[idx_rep - idx_sub]).reshape(X[basin_idx].shape)

        # ensure inputs and target has the right dtype
        self.gages = gages
        self.X = X.astype('float32')
        self.y = flow['B'].values.reshape((flow.shape[0], output_size)).astype('float32')
        self.basin = flow['basin'].values.astype('int32')

        # prevent subclass from calling Dynamic.normalize
        self.norm = None
        if isinstance(self, Dynamic):
            self.normalize()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx], self.basin[idx]]

    # normalization
    def normalize(self, norm=None):
        # cannot normalize twice
        if self.norm is not None:
            return self.norm

        # scaler of X
        if norm is None:
            X_mean = self.X.reshape(-1, self.X.shape[-1]).mean(axis=0)
            X_std = self.X.reshape(-1, self.X.shape[-1]).std(axis=0)
        else:
            X_mean = norm[0]
            X_std = norm[1]

        # scaler of y
        if norm is None:
            y_mean = 0
            y_std = self.y.max(axis=0) - y_mean
        else:
            y_mean = norm[2]
            y_std = norm[3]

        # normalize X and y
        self.X = (self.X - X_mean) / X_std
        self.y = (self.y - y_mean) / y_std
        self.norm = (X_mean, X_std, y_mean, y_std)

        return self.norm
