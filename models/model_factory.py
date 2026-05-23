from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.cnn_model import CNNModel

class ModelFactory:
    """
    Factory class to instantiate various Deep Learning model architectures for time-series.
    """
    @staticmethod
    def get_model(model_name, input_dim, hidden_dim, num_layers, dropout=0.2, window_size=10):
        name = model_name.upper()
        if name == "LSTM":
            return LSTMModel(input_dim, hidden_dim, num_layers, dropout=dropout)
        elif name == "GRU":
            return GRUModel(input_dim, hidden_dim, num_layers, dropout=dropout)
        elif name == "CNN":
            return CNNModel(input_dim, hidden_dim, num_layers, dropout=dropout, window_size=window_size)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

