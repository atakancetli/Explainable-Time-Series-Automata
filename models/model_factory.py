from models.lstm_model import LSTMModel

class ModelFactory:
    @staticmethod
    def get_model(model_name, input_dim, hidden_dim, num_layers):
        if model_name.upper() == "LSTM":
            return LSTMModel(input_dim, hidden_dim, num_layers)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
