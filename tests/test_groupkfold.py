import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.train_utils import set_seed
from utils.data_loader import DataLoader

class TestDeterministicSeedingAndGroupFolding(unittest.TestCase):
    def test_set_seed_reproducibility(self):
        """
        Verifies that set_seed ensures perfect deterministic weight initialization.
        """
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)
            def forward(self, x):
                return self.fc2(self.fc1(x))
                
        # Seed 42 first run
        set_seed(42)
        net1 = SimpleNet()
        w1 = net1.fc1.weight.clone().detach().numpy()
        
        # Seed 42 second run
        set_seed(42)
        net2 = SimpleNet()
        w2 = net2.fc1.weight.clone().detach().numpy()
        
        # Seed 123 run
        set_seed(123)
        net3 = SimpleNet()
        w3 = net3.fc1.weight.clone().detach().numpy()
        
        # w1 and w2 must be identical
        np.testing.assert_array_equal(w1, w2)
        
        # w1 and w3 must be different
        self.assertFalse(np.array_equal(w1, w3))

    def test_groupkfold_no_leakage(self):
        """
        Verifies that the GroupKFold and StratifiedGroupKFold splits strictly prevent
        data leakage by ensuring no source file groups overlap.
        """
        # Create a mock dataframe resembling SKAB data
        data_list = []
        for i in range(1, 6): # 5 files
            df = pd.DataFrame(np.random.randn(20, 2), columns=["sensorA", "sensorB"])
            df['source_file'] = f"file_{i}.csv"
            df['anomaly'] = np.random.choice([0, 1], size=20)
            data_list.append(df)
            
        mock_data = pd.concat(data_list)
        
        loader = DataLoader("SKAB")
        # Generate 3 folds
        folds = loader.split_by_group(mock_data, n_splits=3, stratified=True)
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_files = mock_data.iloc[train_idx]['source_file'].unique()
            val_files = mock_data.iloc[val_idx]['source_file'].unique()
            
            # Intersection of train and validation fold files must be strictly empty
            overlap = set(train_files).intersection(set(val_files))
            self.assertEqual(len(overlap), 0, f"Data leakage detected on fold {fold_idx + 1}! Overlap: {overlap}")

if __name__ == '__main__':
    unittest.main()
