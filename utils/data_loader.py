import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from configs.config import Config

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.config = Config()
        self.scaler = StandardScaler()

    def load_raw_data(self, path):
        pass

    def preprocess(self, data):
        pass

    def split_data(self, data):
        pass
