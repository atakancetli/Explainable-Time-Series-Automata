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
        
        layers = []
        in_channels = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = hidden_dim
            
        self.conv_blocks = nn.Sequential(*layers)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=dropout)
        
        pooled_len = window_size // 2
        
        self.fc = nn.Linear(hidden_dim * pooled_len, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(dim=-1)

