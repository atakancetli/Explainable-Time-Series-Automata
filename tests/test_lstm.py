import unittest
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from models.lstm_model import LSTMModel
from utils.data_loader import TimeSeriesDataset
from utils.train_utils import EarlyStopping

class TestLSTMAndDLBaselines(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory for test checkpoints
        self.test_dir = "test_checkpoints"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Clean up temporary checkpoints directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_lstm_model_dimensions(self):
        # 1. Output dimension checks
        batch_size = 8
        window_size = 10
        input_dim = 5
        hidden_dim = 16
        num_layers = 2
        output_dim = 1
        
        model = LSTMModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=output_dim, 
            dropout=0.2
        )
        
        # Random input batch of shape (batch_size, window_size, input_dim)
        x = torch.randn(batch_size, window_size, input_dim)
        y_pred = model(x)
        
        # The output of standard LSTMModel has sigmoid squeeze so it should be of shape (batch_size,)
        self.assertEqual(y_pred.shape, (batch_size,))
        self.assertTrue((y_pred >= 0.0).all() and (y_pred <= 1.0).all())

    def test_time_series_dataset_validations(self):
        # 2. Sliding dataset loaders & validations
        data = np.arange(100).reshape(50, 2)
        labels = np.ones(50)
        
        # Test valid dataset instantiation
        dataset = TimeSeriesDataset(data, window_size=10, labels=labels)
        self.assertEqual(len(dataset), 40) # 50 - 10 = 40
        
        x, y = dataset[0]
        self.assertEqual(x.shape, (10, 2))
        self.assertEqual(y.item(), 1.0)
        
        # Test without labels
        dataset_no_labels = TimeSeriesDataset(data, window_size=10)
        self.assertEqual(len(dataset_no_labels), 40)
        x_no_labels = dataset_no_labels[0]
        self.assertEqual(x_no_labels.shape, (10, 2))
        
        # Test error: window_size <= 0
        with self.assertRaises(ValueError):
            TimeSeriesDataset(data, window_size=0)
            
        # Test error: data length less than window_size
        with self.assertRaises(ValueError):
            TimeSeriesDataset(data, window_size=100)
            
        # Test error: mismatched label length
        with self.assertRaises(ValueError):
            TimeSeriesDataset(data, window_size=10, labels=labels[:20])

    def test_early_stopping_triggers_and_checkpointing(self):
        # 3. Early stopping triggers & checkpoints
        # Initialize a simple mock linear model to verify state saving/loading
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)
            def forward(self, x):
                return self.linear(x)
                
        model = SimpleModel()
        
        checkpoint_path = os.path.join(self.test_dir, "best_model.pth")
        early_stopping = EarlyStopping(patience=3, delta=0.01, checkpoint_path=checkpoint_path, verbose=False)
        
        # Initially early stop is False, counter is 0
        self.assertFalse(early_stopping.early_stop)
        self.assertEqual(early_stopping.counter, 0)
        
        # Epoch 1: val loss = 0.5 (initial best)
        early_stopping(0.5, model)
        self.assertEqual(early_stopping.best_score, -0.5)
        self.assertEqual(early_stopping.counter, 0)
        self.assertIsNotNone(early_stopping.best_state_dict)
        
        # Epoch 2: val loss = 0.55 (worse, no improvement)
        early_stopping(0.55, model)
        self.assertEqual(early_stopping.counter, 1)
        self.assertFalse(early_stopping.early_stop)
        
        # Epoch 3: val loss = 0.54 (worse, no improvement)
        early_stopping(0.54, model)
        self.assertEqual(early_stopping.counter, 2)
        self.assertFalse(early_stopping.early_stop)
        
        # Epoch 4: val loss = 0.56 (worse, no improvement - patience met)
        early_stopping(0.56, model)
        self.assertEqual(early_stopping.counter, 3)
        self.assertTrue(early_stopping.early_stop)
        
        # Check that checkpoint was saved on disk
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Modify weights slightly
        with torch.no_grad():
            model.linear.weight.fill_(999.0)
            
        # Load best weights and verify they are restored
        early_stopping.load_best_weights(model)
        self.assertNotEqual(model.linear.weight[0, 0].item(), 999.0)

if __name__ == '__main__':
    unittest.main()
