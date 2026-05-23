import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    Standard GRU (Gated Recurrent Unit) model for sequential time-series anomaly detection.
    
    Incorporates dropout regularization and outputs dynamic probability scores.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out).squeeze()
