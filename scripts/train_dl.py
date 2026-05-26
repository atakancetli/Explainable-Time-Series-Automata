import torch
import torch.nn as nn
from configs.config import Config
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader
from utils.train_utils import train_one_epoch, validate, EarlyStopping, save_model, predict, set_seed
from utils.logger import setup_logger
from utils.visualization import plot_loss
import pandas as pd
import numpy as np
import time
import os
from utils.metrics import calculate_metrics, save_results
from sklearn.metrics import f1_score

def find_best_threshold(model, val_loader, val_labels, device):
    """
    Finds the optimal anomaly threshold on the validation set that maximizes the F1-score.
    """
    model.eval()
    y_probs = predict(model, val_loader, device)
    
    # Extract ground truth matching the predictions length (sliding window shift)
    if hasattr(val_loader.dataset, 'labels') and val_loader.dataset.labels is not None:
        y_true = val_loader.dataset.labels[val_loader.dataset.window_size:].cpu().numpy()
    else:
        window_size = len(val_labels) - len(y_probs)
        y_true = val_labels[window_size:]
        
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Search threshold range
    for thresh in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    return float(best_thresh), float(best_f1)

def run_training_skab_kfold(model_type, seed):
    """
    Runs 5-fold Stratified Group K-Fold cross-validation on SKAB for a specific seed.
    Logs metrics, runtimes, and saves fold checkpoints deterministically.
    """
    config = Config()
    set_seed(seed)
    
    logger = setup_logger(f"Train_SKAB_{model_type}_seed{seed}", f"SKAB_{model_type}_seed{seed}.log")
    logger.info(f"--- Starting 5-fold CV for {model_type} on SKAB with Seed {seed} ---")
    
    loader = DataLoader("SKAB")
    raw_data = loader.load_skab(config.SKAB_PATH)
    
    # 5-fold StratifiedGroupKFold splits based on source_file
    folds = loader.split_by_group(raw_data, n_splits=5, stratified=True)
    
    fold_metrics = []
    fold_train_times = []
    fold_inf_times = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"--- Training Fold {fold_idx + 1}/5 ---")
        
        train_df = raw_data.iloc[train_idx]
        val_df = raw_data.iloc[val_idx]
        
        # Fit scaler + PCA on training fold ONLY to prevent data leakage
        loader_fold = DataLoader("SKAB")
        train_scaled, _ = loader_fold.preprocess(train_df, fit_scaler=True)
        val_scaled, _ = loader_fold.transform(val_df)
        
        train_labels = loader_fold.get_labels(train_df)
        val_labels = loader_fold.get_labels(val_df)
        
        train_loader, val_loader = loader_fold.get_fold_dataloaders(
            train_scaled, val_scaled, train_labels, val_labels
        )
        
        # Initialize deterministic model under current seed
        set_seed(seed)
        model = ModelFactory.get_model(
            model_type, 
            train_scaled.shape[1], 
            config.HIDDEN_SIZE, 
            config.NUM_LAYERS,
            dropout=0.2,
            window_size=config.WINDOW_SIZE
        ).to(config.DEVICE)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        checkpoint_path = f"checkpoints/SKAB_{model_type}_seed{seed}_fold{fold_idx+1}_best.pth"
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            checkpoint_path=checkpoint_path,
            verbose=False
        )
        
        # Train fold & record training time
        start_train = time.time()
        for epoch in range(config.MAX_EPOCHS):
            train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            val_loss = validate(model, val_loader, criterion, config.DEVICE)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        end_train = time.time()
        train_time = end_train - start_train
        fold_train_times.append(train_time)
        
        # Load best fold checkpoint
        early_stopping.load_best_weights(model)
        
        # Evaluate validation inference & record inference time
        start_inf = time.time()
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        end_inf = time.time()
        inf_time = end_inf - start_inf
        fold_inf_times.append(inf_time)
        
        # Threshold optimization on fold validation set
        best_thresh, _ = find_best_threshold(model, val_loader, val_labels, config.DEVICE)
        
        # Metrics calculation on fold validation
        model.eval()
        y_probs = predict(model, val_loader, config.DEVICE)
        window_size = len(val_labels) - len(y_probs)
        y_true = val_labels[window_size:]
        y_pred = (y_probs >= best_thresh).astype(int)
        
        metrics = calculate_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        
        logger.info(f"Fold {fold_idx+1} complete: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}, Train Time={train_time:.2f}s")
        
    # Aggregate metrics across the 5 folds
    avg_f1 = np.mean([m["f1"] for m in fold_metrics])
    avg_prec = np.mean([m["precision"] for m in fold_metrics])
    avg_rec = np.mean([m["recall"] for m in fold_metrics])
    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    avg_train_time = np.mean(fold_train_times)
    avg_inf_time = np.mean(fold_inf_times)
    
    results = {
        "dataset": "SKAB",
        "model": model_type,
        "seed": seed,
        "f1": float(avg_f1),
        "precision": float(avg_prec),
        "recall": float(avg_rec),
        "accuracy": float(avg_acc),
        "training_time_sec": float(avg_train_time),
        "inference_time_sec": float(avg_inf_time)
    }
    
    save_results(results, "dl_metrics.csv")
    logger.info(f"--- Seed {seed} Overall Results: F1={avg_f1:.4f}, Accuracy={avg_acc:.4f}, Avg Train Time={avg_train_time:.2f}s ---")
    return results

def run_training_batadal_chronological(model_type, seed):
    """
    Runs chronological split training (60/20/20) on BATADAL for a specific seed.
    Logs metrics, runtimes, and saves best checkpoint deterministically.
    """
    config = Config()
    set_seed(seed)
    
    logger = setup_logger(f"Train_BATADAL_{model_type}_seed{seed}", f"BATADAL_{model_type}_seed{seed}.log")
    logger.info(f"--- Starting Chronological Training for {model_type} on BATADAL with Seed {seed} ---")
    
    loader = DataLoader("BATADAL")
    path = config.BATADAL_PATH
    if os.path.isdir(path):
        path = os.path.join(path, "batadal_training_2.csv")
        
    raw_data = loader.load_batadal(path)
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    train_data, val_data, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    train_labels, val_labels, test_labels = loader.split_chronological(pd.Series(labels))
    
    train_loader, val_loader, test_loader = loader.get_dataloaders(
        train_data.values, val_data.values, test_data.values,
        train_labels.values, val_labels.values, test_labels.values
    )
    
    # Initialize deterministic model under current seed
    set_seed(seed)
    model = ModelFactory.get_model(
        model_type, 
        scaled_data.shape[1], 
        config.HIDDEN_SIZE, 
        config.NUM_LAYERS,
        dropout=0.2,
        window_size=config.WINDOW_SIZE
    ).to(config.DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    checkpoint_path = f"checkpoints/BATADAL_{model_type}_seed{seed}_best.pth"
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        checkpoint_path=checkpoint_path,
        verbose=False
    )
    
    # Train model & record training time
    start_train = time.time()
    for epoch in range(config.MAX_EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    end_train = time.time()
    train_time = end_train - start_train
    
    # Load best checkpoint weights
    early_stopping.load_best_weights(model)
    
    # Evaluate validation inference & record inference time
    start_inf = time.time()
    val_loss = validate(model, val_loader, criterion, config.DEVICE)
    end_inf = time.time()
    inf_time = end_inf - start_inf
    
    # Threshold optimization on validation set
    best_thresh, _ = find_best_threshold(model, val_loader, val_labels.values, config.DEVICE)
    
    # Predict and compute metrics
    model.eval()
    y_probs = predict(model, val_loader, config.DEVICE)
    window_size = len(val_labels) - len(y_probs)
    y_true = val_labels.values[window_size:]
    y_pred = (y_probs >= best_thresh).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred)
    
    results = {
        "dataset": "BATADAL",
        "model": model_type,
        "seed": seed,
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "accuracy": float(metrics["accuracy"]),
        "training_time_sec": float(train_time),
        "inference_time_sec": float(inf_time)
    }
    
    save_results(results, "dl_metrics.csv")
    logger.info(f"BATADAL complete: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}, Train Time={train_time:.2f}s")
    return results

if __name__ == "__main__":
    seeds = [42, 123, 2026, 7, 999]
    for dataset in ["SKAB", "BATADAL"]:
        for m in ["LSTM", "GRU", "CNN"]:
            for seed in seeds:
                try:
                    if dataset == "SKAB":
                        run_training_skab_kfold(m, seed)
                    else:
                        run_training_batadal_chronological(m, seed)
                except Exception as e:
                    print(f"Error training {m} on {dataset} with seed {seed}: {e}")
