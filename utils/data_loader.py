import pandas as pd
import numpy as np
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from configs.config import Config
from typing import Optional, Tuple, Union

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapper for time-series data using sliding window chunking.
    
    This dataset constructs sequences of length `window_size` from sequential
    sensor readings to prepare them for RNN/LSTM model consumption.
    
    Attributes:
        data (torch.Tensor): Model feature tensor of shape (N, num_features).
        window_size (int): Size of the temporal sliding window.
        labels (Optional[torch.Tensor]): Target anomaly binary label tensor of shape (N,).
    """
    def __init__(
        self, 
        data: Union[np.ndarray, pd.DataFrame, torch.Tensor], 
        window_size: int, 
        labels: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None
    ) -> None:
        """
        Initializes the sliding window dataset.
        
        Args:
            data: Feature sequences, either as numpy array, pandas DataFrame, or PyTorch tensor.
            window_size: Length of each feature sequence window.
            labels: Optional labels mapping to each step in the dataset.
            
        Raises:
            ValueError: If window_size is non-positive or exceeds the total data length.
            ValueError: If the length of data and labels does not match.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be a positive integer. Got {window_size}")
            
        if len(data) < window_size:
            raise ValueError(
                f"Data length ({len(data)}) must be greater than or equal to window_size ({window_size})."
            )
            
        if labels is not None and len(labels) != len(data):
            raise ValueError(
                f"Length of data ({len(data)}) and labels ({len(labels)}) must match."
            )

        # Convert data safely to PyTorch tensor
        if isinstance(data, torch.Tensor):
            self.data = data.clone().detach().float()
        elif isinstance(data, (np.ndarray, pd.DataFrame)):
            val = data.values if isinstance(data, pd.DataFrame) else data
            self.data = torch.tensor(val, dtype=torch.float32)
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
            
        self.window_size = window_size
        
        # Convert labels safely to PyTorch tensor
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                self.labels = labels.clone().detach().float()
            elif isinstance(labels, (np.ndarray, pd.Series)):
                val_lbl = labels.values if isinstance(labels, pd.Series) else labels
                self.labels = torch.tensor(val_lbl, dtype=torch.float32)
            else:
                self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self) -> int:
        """
        Returns the total number of sliding window sequences.
        """
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieves the sequence block at the given index.
        
        Args:
            idx: Base starting index for the sliding window.
            
        Returns:
            A tuple of (x, y) if labels are provided, where x is of shape (window_size, num_features)
            and y is the target scalar. Otherwise, returns only the feature sequence x.
        """
        max_idx = len(self) - 1
        if idx < 0 or idx > max_idx:
            raise IndexError(f"Index {idx} out of range for TimeSeriesDataset of length {len(self)}")
            
        x = self.data[idx : idx + self.window_size]
        if self.labels is not None:
            y = self.labels[idx + self.window_size]
            return x, y
        return x


class DataLoader:
    """
    DataLoader handles dataset retrieval, formatting, and preprocessing pipelines
    for the SKAB and BATADAL time-series anomaly detection datasets.
    
    It incorporates strict data-leakage prevention strategies by dividing
    normalization and PCA fitting to the training fold only, as well as providing
    advanced stratified group K-fold cross-validation based on source files.
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.config = Config()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.PCA_COMPONENTS)
        
        # Configure logging for data processing
        self.logger = logging.getLogger(f"DataLoader_{dataset_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)


    def load_skab(self, path):
        subdirs = ['valve1', 'valve2']
        data_list = []
        
        self.logger.info(f"Scanning SKAB path: {path}")
        
        # Flag to check if we loaded any data
        loaded_from_subdirs = False
        
        for subdir in subdirs:
            subdir_path = os.path.join(path, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                self.logger.info(f"Loading data from subdirectory: {subdir}")
                all_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.csv')]
                for f in all_files:
                    try:
                        df = pd.read_csv(f, sep=';', index_col='datetime', parse_dates=True)
                        df['source_group'] = subdir
                        df['source_file'] = os.path.basename(f)
                        data_list.append(df)
                        self.logger.info(f"Loaded {os.path.basename(f)}: shape {df.shape}")
                        loaded_from_subdirs = True
                    except Exception as e:
                        self.logger.error(f"Error loading {f}: {e}")
            else:
                self.logger.warning(f"Subdirectory not found: {subdir_path}")
                
        # Fallback: if no subdirs found or empty, search directly in the main path
        if not loaded_from_subdirs:
            self.logger.info(f"Attempting fallback to scan files directly in root SKAB path: {path}")
            if os.path.exists(path):
                all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
                for f in all_files:
                    try:
                        df = pd.read_csv(f, sep=';', index_col='datetime', parse_dates=True)
                        df['source_group'] = 'root'
                        df['source_file'] = os.path.basename(f)
                        data_list.append(df)
                        self.logger.info(f"Loaded fallback file {os.path.basename(f)}: shape {df.shape}")
                    except Exception as e:
                        self.logger.error(f"Error loading fallback file {f}: {e}")
                    
        if not data_list:
            raise FileNotFoundError(f"No SKAB CSV files found under {path} or its subdirectories.")
            
        combined_df = pd.concat(data_list)
        self.logger.info(f"Successfully concatenated SKAB dataset. Combined shape: {combined_df.shape}")
        return combined_df


    def load_batadal(self, path):
        self.logger.info(f"Loading BATADAL dataset from {path}")
        if not os.path.exists(path):
            self.logger.warning(f"BATADAL dataset path not found at: {path}")
            raise FileNotFoundError(f"BATADAL dataset file not found at {path}")
            
        # Load CSV (flexible separator)
        df = pd.read_csv(path, sep=None, engine='python')
        self.logger.info(f"Loaded raw file. Initial shape: {df.shape}")
        
        # 1. Datetime index resolution
        dt_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if dt_cols:
            self.logger.info(f"Setting index to datetime column: {dt_cols[0]}")
            df[dt_cols[0]] = pd.to_datetime(df[dt_cols[0]], errors='coerce')
            df.set_index(dt_cols[0], inplace=True)
            df.index.name = 'datetime'
        else:
            self.logger.warning("No datetime column found. Ensuring index is standard range index.")
            
        # 2. Target label mapping
        label_cols = [c for c in df.columns if c in ['ATT_FLAG', 'anomaly', 'label', 'Label', 'class'] or 'attack' in c.lower() or 'flag' in c.lower()]
        if label_cols:
            self.logger.info(f"Mapping column '{label_cols[0]}' to standard target label 'anomaly'")
            df['anomaly'] = df[label_cols[0]].astype(int)
            # Remove original if renamed
            if label_cols[0] != 'anomaly':
                df.drop(columns=[label_cols[0]], inplace=True)
        else:
            self.logger.warning("No standard BATADAL attack flag/anomaly label column found. Initializing default 'anomaly' column to 0.")
            df['anomaly'] = 0
            
        # Clean whitespaces in column names if any
        df.columns = [c.strip() for c in df.columns]
        
        self.logger.info(f"Successfully processed BATADAL dataset. Final shape: {df.shape}")
        return df



    def _get_features(self, data):
        # 1. Standard exclusion list
        drop_cols = ['anomaly', 'changepoint', 'source_group', 'source_file', 'datetime', 'DATETIME', 'ATT_FLAG']
        features = data.drop(columns=[col for col in drop_cols if col in data.columns], errors='ignore')
        
        # 2. Exclude any datetime indices or columns that have date/time in their name
        time_related_cols = [col for col in features.columns if 'time' in col.lower() or 'date' in col.lower() or 'day' in col.lower()]
        if time_related_cols:
            self.logger.info(f"Automatically excluding time-related feature columns: {time_related_cols}")
            features = features.drop(columns=time_related_cols)
            
        # 3. Exclude non-numeric object/categorical columns just in case
        non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            self.logger.info(f"Automatically excluding non-numeric feature columns: {non_numeric_cols}")
            features = features.drop(columns=non_numeric_cols)
            
        return features


    def fit(self, train_data):
        self.logger.info("Fitting StandardScaler and PCA on training data to prevent data leakage...")
        features = self._get_features(train_data)
        scaled_features = self.scaler.fit_transform(features)
        self.pca.fit(scaled_features)
        self.logger.info(f"Fit completed successfully. PCA features: {self.pca.n_components_}")
        return self

    def transform(self, data):
        features = self._get_features(data)
        scaled_data = self.scaler.transform(features)
        pc1 = self.pca.transform(scaled_data)
        return scaled_data, pc1

    def preprocess(self, data, fit_scaler=True):
        if fit_scaler:
            self.fit(data)
        return self.transform(data)


    def get_labels(self, data):
        if 'anomaly' in data.columns:
            return data['anomaly'].values
        return None

    def split_chronological(self, data):
        self.logger.info("Executing chronological split of the dataset...")
        
        # Verify split ratios
        ratios_sum = self.config.TRAIN_RATIO + self.config.VAL_RATIO + self.config.TEST_RATIO
        if abs(ratios_sum - 1.0) > 1e-5:
            raise ValueError(f"Train/Val/Test split ratios must sum to 1.0. Current ratios: Train={self.config.TRAIN_RATIO}, Val={self.config.VAL_RATIO}, Test={self.config.TEST_RATIO}")
            
        # Verify index chronological ordering
        if isinstance(data.index, pd.DatetimeIndex):
            is_sorted = data.index.is_monotonic_increasing
            if not is_sorted:
                self.logger.warning("Dataset index is NOT chronologically sorted! Sorting it now to preserve temporal order.")
                data = data.sort_index()
            else:
                self.logger.info("Dataset index is confirmed to be chronologically sorted.")
                
        n = len(data)
        train_end = int(n * self.config.TRAIN_RATIO)
        val_end = train_end + int(n * self.config.VAL_RATIO)
        
        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]
        
        self.logger.info(f"Chronological split complete: Train={train.shape[0]} ({self.config.TRAIN_RATIO*100:.0f}%), Val={val.shape[0]} ({self.config.VAL_RATIO*100:.0f}%), Test={test.shape[0]} ({self.config.TEST_RATIO*100:.0f}%)")
        return train, val, test


    def get_dataloaders(self, train_data, val_data, test_data, train_labels, val_labels, test_labels):
        train_ds = TimeSeriesDataset(train_data, self.config.WINDOW_SIZE, train_labels)
        val_ds = TimeSeriesDataset(val_data, self.config.WINDOW_SIZE, val_labels)
        test_ds = TimeSeriesDataset(test_data, self.config.WINDOW_SIZE, test_labels)
        
        train_loader = TorchDataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = TorchDataLoader(val_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = TorchDataLoader(test_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def split_by_group(self, data, n_splits=5, stratified=True):
        self.logger.info(f"Splitting data using {'StratifiedGroupKFold' if stratified else 'GroupKFold'} with {n_splits} splits.")
        if stratified:
            try:
                gkf = StratifiedGroupKFold(n_splits=n_splits)
            except Exception as e:
                self.logger.warning(f"Error initializing StratifiedGroupKFold: {e}. Falling back to GroupKFold.")
                gkf = GroupKFold(n_splits=n_splits)
        else:
            gkf = GroupKFold(n_splits=n_splits)
            
        groups = data['source_file']
        features = self._get_features(data)
        labels = data['anomaly'] if 'anomaly' in data.columns else np.zeros(len(data))
        
        splits = []
        for train_idx, test_idx in gkf.split(features, labels, groups=groups):
            splits.append((train_idx, test_idx))
            
        self.logger.info(f"Data split completed. Generated {len(splits)} folds.")
        return splits

