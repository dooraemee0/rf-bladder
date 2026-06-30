import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

def load_full_dataframe(path):
    return pd.read_csv(path)

class PreprocessTransform:
    def __init__(self, num_select, sigma=1, downsample_rate=10, max_len=400, seed_offset=0):
        self.num_select = num_select
        self.sigma = sigma
        self.downsample_rate = downsample_rate
        self.max_len = max_len
        self.seed_offset = seed_offset

    def __call__(self, data, seed):
        rng = np.random.RandomState(seed + self.seed_offset)

        envelope = np.abs(hilbert(data, axis=1))
        smoothed = gaussian_filter1d(envelope, sigma=self.sigma, axis=1)
        smoothed = smoothed[::self.downsample_rate, :][:self.max_len, :]

        mean, std = smoothed.mean(), smoothed.std()
        norm_data = (smoothed - mean) / (std + 1e-8)

        num_columns = norm_data.shape[1]
        group_size = max(1, num_columns // self.num_select)
        selected = []
        for i in range(self.num_select):
            start = i * group_size
            end = start + group_size if i < self.num_select - 1 else num_columns
            col_idx = rng.choice(range(start, end))
            selected.append(norm_data[:, col_idx])

        return np.stack(selected)

class FolderDataset(Dataset):
    def __init__(self, dataframe, folder_path, transform_h=None, transform_s=None, max_depth=4000):
        self.database = dataframe
        self.folder_path = folder_path
        self.transform_h = transform_h
        self.transform_s = transform_s
        self.max_depth = max_depth

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        # Use iloc since reset_index() was called in train.py
        row = self.database.iloc[idx]

        # glob.glob returns a list, so use the first element
        h_file = glob.glob(os.path.join(self.folder_path, row['Upright_H'] + '*.csv'))[0]
        s_file = glob.glob(os.path.join(self.folder_path, row['Upright_S'] + '*.csv'))[0]

        h_df = pd.read_csv(h_file, header=None, low_memory=False)
        h_data = h_df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()[:self.max_depth, :]

        s_df = pd.read_csv(s_file, header=None, low_memory=False)
        s_data = s_df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()[:self.max_depth, :]

        # Use the original index saved in train.py
        original_index = row['original_index']

        # Use a random seed for different augmentations on each call
        seed = np.random.randint(0, 100000) 

        if self.transform_h:
            h_data = self.transform_h(h_data, seed=seed)
        if self.transform_s:
            s_data = self.transform_s(s_data, seed=seed)

        sample = {
            'horizon': torch.tensor(h_data, dtype=torch.float32),
            'sagittal': torch.tensor(s_data, dtype=torch.float32),
            'volume_gt': torch.tensor(row['volume'], dtype=torch.float32),
            'index': original_index
        }

        return sample