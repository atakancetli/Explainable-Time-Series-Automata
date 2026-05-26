import torch
import pandas as pd
import os
from configs.config import Config
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader
from utils.train_utils import load_model, predict
from utils.metrics import calculate_metrics, save_results

def evaluate_model_batadal_multi_seed(model_type, seed):
    config = Config()
    loader = DataLoader("BATADAL")
    path = config.BATADAL_PATH
    if os.path.isdir(path):
        path = os.path.join(path, "batadal_training_2.csv")
    raw_data = loader.load_batadal(path)
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    train_data, val_data, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    train_labels, val_labels, test_labels = loader.split_chronological(pd.Series(labels))
    
    _, val_loader, test_loader = loader.get_dataloaders(
        train_data.values, val_data.values, test_data.values,
        train_labels.values, val_labels.values, test_labels.values
    )
    
    model = ModelFactory.get_model(
        model_type, 
        scaled_data.shape[1], 
        config.HIDDEN_SIZE, 
        config.NUM_LAYERS,
        dropout=0.2,
        window_size=config.WINDOW_SIZE
    ).to(config.DEVICE)
    
    checkpoint_path = f"checkpoints/BATADAL_{model_type}_seed{seed}_best.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    model = load_model(model, checkpoint_path, config.DEVICE)
    model.eval()
    
    from scripts.train_dl import find_best_threshold
    best_thresh, _ = find_best_threshold(model, val_loader, val_labels.values, config.DEVICE)
    
    y_probs = predict(model, test_loader, config.DEVICE)
    window_size = len(test_labels) - len(y_probs)
    y_true = test_labels.values[window_size:]
    y_pred = (y_probs >= best_thresh).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred)
    return metrics

if __name__ == "__main__":
    seeds = [42, 123, 2026, 7, 999]
    for m in ["LSTM", "GRU", "CNN"]:
        for seed in seeds:
            try:
                metrics = evaluate_model_batadal_multi_seed(m, seed)
                print(f"Results for BATADAL {m} (Seed {seed}): {metrics}")
            except Exception as e:
                print(f"Error evaluating {m} on BATADAL with seed {seed}: {e}")

