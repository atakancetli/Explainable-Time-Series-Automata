from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel

class ModelFactory:
    @staticmethod
    def get_model(model_name, input_dim, hidden_dim, num_layers):
        if model_name.upper() == "LSTM":
            return LSTMModel(input_dim, hidden_dim, num_layers)
        elif model_name.upper() == "CNN":
            return CNNModel(input_dim, hidden_dim, num_layers)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
