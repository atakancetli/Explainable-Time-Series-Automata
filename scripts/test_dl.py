import torch
import pandas as pd
import os
from configs.config import Config
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader
from utils.train_utils import load_model, predict
from utils.metrics import calculate_metrics, save_results

def evaluate_model(dataset_name, model_type):
    config = Config()
    loader = DataLoader(dataset_name)
    if dataset_name == "SKAB":
        raw_data = loader.load_skab(config.SKAB_PATH)
    else:
        path = config.BATADAL_PATH
        if os.path.isdir(path):
            path = os.path.join(path, "batadal_training_2.csv")
        raw_data = loader.load_batadal(path)
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    _, _, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    _, _, test_labels = loader.split_chronological(pd.Series(labels))
    
    _, _, test_loader = loader.get_dataloaders(
        None, None, test_data.values,
        None, None, test_labels.values
    )
    
    model = ModelFactory.get_model(
        model_type, 
        scaled_data.shape[1], 
        config.HIDDEN_SIZE, 
        config.NUM_LAYERS,
        dropout=0.2,
        window_size=config.WINDOW_SIZE
    ).to(config.DEVICE)
    model = load_model(model, f"checkpoints/{dataset_name}_{model_type}.pth", config.DEVICE)
    
    y_pred = predict(model, test_loader, config.DEVICE)
    # Calculate comprehensive evaluation metrics (Accuracy, Precision, Recall, and F1-score)
    metrics = calculate_metrics(test_labels.values[config.WINDOW_SIZE:], (y_pred > 0.5).astype(int))
    
    results = {"dataset": dataset_name, "model": model_type, **metrics}
    save_results(results, "final_metrics.csv")
    print(f"Results for {model_type} on {dataset_name}: {metrics}")

if __name__ == "__main__":
    for dataset in ["SKAB", "BATADAL"]:
        for m in ["LSTM", "GRU", "CNN"]:
            try:
                evaluate_model(dataset, m)
            except Exception as e:
                print(f"Error evaluating {m} on {dataset}: {e}")

