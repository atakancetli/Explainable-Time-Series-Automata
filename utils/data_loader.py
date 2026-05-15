import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from configs.config import Config

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

    def preprocess(self, data):
        pass

    def split_data(self, data):
        pass
