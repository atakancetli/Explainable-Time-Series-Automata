import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    Standard 1D Convolutional Neural Network model for sequential time-series anomaly detection.
    
    Generalizes sequence-length pooling to support arbitrary window sizes, preventing
    dimension mismatches during parameter sensitivity sweeps.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2, window_size=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=dropout)
        
        # Calculate pooled output sequence length dynamically based on window size
        pooled_len = window_size // 2
        
        self.fc = nn.Linear(hidden_dim * pooled_len, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, window_size, input_dim)
        # Convert to: (batch_size, input_dim, window_size) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x).squeeze()

