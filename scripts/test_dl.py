import torch
import pandas as pd
import numpy as np
import os
from configs.config import Config
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader
from utils.train_utils import load_model, predict
from utils.metrics import calculate_metrics, save_results

def evaluate_model_skab_kfold(model_type, seed):
    config = Config()
    loader = DataLoader("SKAB")
    raw_data = loader.load_skab(config.SKAB_PATH)
    
    folds = loader.split_by_group(raw_data, n_splits=4, stratified=True)
    
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_df = raw_data.iloc[train_idx]
        val_df = raw_data.iloc[val_idx]
        
        loader_fold = DataLoader("SKAB")
        train_scaled, _ = loader_fold.preprocess(train_df, fit_scaler=True)
        val_scaled, _ = loader_fold.transform(val_df)
        train_labels = loader_fold.get_labels(train_df)
        val_labels = loader_fold.get_labels(val_df)
        
        _, val_loader = loader_fold.get_fold_dataloaders(
            train_scaled, val_scaled, train_labels, val_labels
        )
        
        model = ModelFactory.get_model(
            model_type, 
            val_scaled.shape[1], 
            config.HIDDEN_SIZE, 
            config.NUM_LAYERS,
            dropout=0.2,
            window_size=config.WINDOW_SIZE
        ).to(config.DEVICE)
        
        checkpoint_path = f"checkpoints/SKAB_{model_type}_seed{seed}_fold{fold_idx+1}_best.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        model = load_model(model, checkpoint_path, config.DEVICE)
        model.eval()
        
        from scripts.train_dl import find_best_threshold
        best_thresh, _ = find_best_threshold(model, val_loader, val_labels, config.DEVICE)
        
        y_probs = predict(model, val_loader, config.DEVICE)
        window_size = len(val_labels) - len(y_probs)
        y_true = val_labels[window_size:]
        y_pred = (y_probs >= best_thresh).astype(int)
        
        metrics = calculate_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        
    avg_f1 = np.mean([m["f1"] for m in fold_metrics])
    avg_prec = np.mean([m["precision"] for m in fold_metrics])
    avg_rec = np.mean([m["recall"] for m in fold_metrics])
    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    
    return {
        "f1": float(avg_f1),
        "precision": float(avg_prec),
        "recall": float(avg_rec),
        "accuracy": float(avg_acc)
    }

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
    config = Config()
    seeds = config.SEEDS
    metrics_file = "dl_test_metrics.csv"
    
    # Remove previous test metrics file if it exists to have clean results
    os.makedirs("results/metrics", exist_ok=True)
    path = os.path.join("results/metrics", metrics_file)
    if os.path.exists(path):
        os.remove(path)
        
    for dataset in ["SKAB", "BATADAL"]:
        for m in ["LSTM", "GRU", "CNN"]:
            for seed in seeds:
                try:
                    if dataset == "SKAB":
                        metrics = evaluate_model_skab_kfold(m, seed)
                    else:
                        metrics = evaluate_model_batadal_multi_seed(m, seed)
                        
                    results = {
                        "dataset": dataset,
                        "model": m,
                        "seed": seed,
                        **metrics
                    }
                    save_results(results, metrics_file)
                    print(f"Results for {dataset} {m} (Seed {seed}): {metrics}")
                except Exception as e:
                    print(f"Error evaluating {m} on {dataset} with seed {seed}: {e}")

