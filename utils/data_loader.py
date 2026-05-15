import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from configs.config import Config

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        if self.labels is not None:
            y = self.labels[idx + self.window_size]
            return x, y
        return x

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.config = Config()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.PCA_COMPONENTS)

    def load_skab(self, path):
        all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        data_list = []
        for f in all_files:
            df = pd.read_csv(f, sep=';', index_col='datetime', parse_dates=True)
            df['source_file'] = os.path.basename(f)
            data_list.append(df)
        return pd.concat(data_list)

    def load_batadal(self, path):
        df = pd.read_csv(path, index_col='DATETIME', parse_dates=True)
        return df

    def preprocess(self, data, fit_scaler=True):
        features = data.drop(['anomaly', 'changepoint', 'source_file'], axis=1, errors='ignore')
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(features)
        else:
            scaled_data = self.scaler.transform(features)
        
        pc1 = self.pca.fit_transform(scaled_data)
        return scaled_data, pc1

    def get_labels(self, data):
        if 'anomaly' in data.columns:
            return data['anomaly'].values
        return None

    def split_chronological(self, data):
        n = len(data)
        train_end = int(n * self.config.TRAIN_RATIO)
        val_end = train_end + int(n * self.config.VAL_RATIO)
        
        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]
        return train, val, test

    def get_dataloaders(self, train_data, val_data, test_data, train_labels, val_labels, test_labels):
        train_ds = TimeSeriesDataset(train_data, self.config.WINDOW_SIZE, train_labels)
        val_ds = TimeSeriesDataset(val_data, self.config.WINDOW_SIZE, val_labels)
        test_ds = TimeSeriesDataset(test_data, self.config.WINDOW_SIZE, test_labels)
        
        train_loader = TorchDataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = TorchDataLoader(val_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = TorchDataLoader(test_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def split_by_group(self, data, n_splits=5):
        gkf = GroupKFold(n_splits=n_splits)
        groups = data['source_file']
        features = data.drop(['anomaly', 'changepoint', 'source_file'], axis=1, errors='ignore')
        labels = data['anomaly']
        
        splits = []
        for train_idx, test_idx in gkf.split(features, labels, groups=groups):
            splits.append((train_idx, test_idx))
        return splits
