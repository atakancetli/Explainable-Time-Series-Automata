import torch
import pandas as pd
from configs.config import Config
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader
from utils.train_utils import load_model, predict
from utils.metrics import calculate_metrics, save_results

def evaluate_model(dataset_name, model_type):
    config = Config()
    loader = DataLoader(dataset_name)
    raw_data = loader.load_skab(config.SKAB_PATH) if dataset_name == "SKAB" else loader.load_batadal(config.BATADAL_PATH)
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    _, _, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    _, _, test_labels = loader.split_chronological(pd.Series(labels))
    
    _, _, test_loader = loader.get_dataloaders(
        None, None, test_data.values,
        None, None, test_labels.values
    )
    
    model = ModelFactory.get_model(model_type, scaled_data.shape[1], config.HIDDEN_SIZE, config.NUM_LAYERS).to(config.DEVICE)
    model = load_model(model, f"checkpoints/{dataset_name}_{model_type}.pth", config.DEVICE)
    
    y_pred = predict(model, test_loader, config.DEVICE)
    metrics = calculate_metrics(test_labels.values[config.WINDOW_SIZE:], (y_pred > 0.5).astype(int))
    
    results = {"dataset": dataset_name, "model": model_type, **metrics}
    save_results(results, "final_metrics.csv")
    print(f"Results for {model_type} on {dataset_name}: {metrics}")

if __name__ == "__main__":
    evaluate_model("SKAB", "LSTM")
