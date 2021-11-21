import os
import numpy as np
import pandas as pd
from src.datasets.dynamic import Dynamic


class WithStatic(Dynamic):
    # load the dataset
    def __init__(self, gages_dir, S_dir, X_dir, y_dir, beg_year, beg_month,
                 seq_length, input_size_sta, input_size_dyn, output_size, eco=None):
        super().__init__(gages_dir, X_dir, y_dir, beg_year, beg_month,
                         seq_length, input_size_dyn, output_size, eco=eco)
        df_sta = pd.read_csv(S_dir, dtype={'STAID': str})
        S = pd.merge(self.gages, df_sta, how='left', on='STAID').iloc[:, -input_size_sta:]
        self.S = S.values[self.basin].astype('float32')

        # normalization static
        self.normalize()

    def __getitem__(self, idx):
        return [(self.S[idx], self.X[idx]), self.y[idx], self.basin[idx]]

    def normalize(self, norm=None):
        # cannot normalize twice
        if self.norm is not None and len(self.norm) == 6:
            return self.norm

        if self.norm is None:
            if norm is None:
                super().normalize()
            else:
                super().normalize(norm=norm[2:])

        if hasattr(self, 'S'):
            if norm is None:
                S_mean = self.S.mean(axis=0)
                S_std = self.S.std(axis=0)
            else:
                S_mean = norm[0]
                S_std = norm[1]
            self.S = (self.S - S_mean) / S_std
            self.norm = (S_mean, S_std, *self.norm)

        return self.norm
