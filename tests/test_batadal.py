import unittest
import numpy as np
import pandas as pd
import torch
from utils.data_loader import DataLoader
from utils.train_utils import set_seed
from scripts.train_dl import find_best_threshold
from models.lstm_model import LSTMModel

class TestBATADALChronologicalSweeps(unittest.TestCase):
    def setUp(self):
        # Create a mock BATADAL dataframe for split tests
        data_list = []
        date_range = pd.date_range(start="2026-05-01", periods=100, freq="H")
        self.mock_data = pd.DataFrame({
            "datetime": date_range,
            "sensor1": np.random.randn(100),
            "sensor2": np.random.randn(100),
            "ATT_FLAG": np.random.choice([0, 1], size=100)
        })
        self.mock_data.set_index("datetime", inplace=True)

    def test_chronological_splits_ratios(self):
        """
        Verifies that split_chronological strictly divides the data
        according to the Config's Train/Val/Test ratios (60/20/20).
        """
        loader = DataLoader("BATADAL")
        
        train, val, test = loader.split_chronological(self.mock_data)
        
        n = len(self.mock_data)
        self.assertEqual(len(train), int(n * 0.6))
        self.assertEqual(len(val), int(n * 0.2))
        self.assertEqual(len(test), n - len(train) - len(val))
        
        # Ensure chronological ordering is preserved (train index < val index < test index)
        self.assertTrue(train.index.max() < val.index.min())
        self.assertTrue(val.index.max() < test.index.min())

    def test_threshold_optimizer_deterministic(self):
        """
        Verifies find_best_threshold behaves deterministically under deterministic seeding.
        """
        set_seed(42)
        model = LSTMModel(input_dim=2, hidden_dim=8, num_layers=1)
        
        # Create mock validation data loader
        val_data = np.random.randn(30, 2)
        val_labels = np.random.choice([0, 1], size=30)
        
        loader = DataLoader("BATADAL")
        _, val_loader, _ = loader.get_dataloaders(
            val_data, val_data, val_data,
            val_labels, val_labels, val_labels
        )
        
        # Get threshold and F1 score first run
        set_seed(42)
        thresh1, f1_1 = find_best_threshold(model, val_loader, val_labels, "cpu")
        
        # Get threshold and F1 score second run
        set_seed(42)
        thresh2, f1_2 = find_best_threshold(model, val_loader, val_labels, "cpu")
        
        self.assertEqual(thresh1, thresh2)
        self.assertEqual(f1_1, f1_2)

if __name__ == '__main__':
    unittest.main()
