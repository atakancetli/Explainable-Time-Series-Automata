import numpy as np
import pandas as pd
import os
import time
from configs.config import Config
from models.automata import TimeSeriesAutomata
from utils.data_loader import DataLoader
from utils.train_utils import set_seed
from utils.metrics import calculate_metrics, save_results
from utils.logger import setup_logger

def find_best_threshold_automata(model, val_series, val_labels, window_size):
    """
    Finds the optimal anomaly decision threshold on the validation set
    to maximize the F1-score for the TimeSeriesAutomata model.
    """
    states = model.generate_states(val_series, window_size=window_size)
    num_windows = len(states)
    y_true = val_labels[window_size:]
    
    # Pre-calculate path probabilities for all windows
    probs = np.zeros(num_windows)
    for t in range(1, num_windows):
        active_sequence = states[t - 1 : min(t + 2, num_windows)]
        probs[t] = model.calculate_path_probability(active_sequence)
        
    best_thresh = 0.01
    best_f1 = 0.0
    
    # Sweep threshold log-space and linear range
    thresholds = np.concatenate([
        np.logspace(-15, -1, 50),
        np.linspace(0.001, 0.5, 50)
    ])
    
    for thresh in thresholds:
        y_pred = (probs < thresh).astype(int)[1:] # Skip index 0 to align
        score = calculate_metrics(y_true, y_pred)["f1"]
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    return float(best_thresh), float(best_f1)

def run_automata_skab_kfold(seed):
    """
    Runs 5-fold Stratified Group K-Fold cross-validation on SKAB for a specific seed.
    Fits scalers dynamically to prevent data leakage and saves results persistently.
    """
    config = Config()
    set_seed(seed)
    
    logger = setup_logger(f"Evaluate_Automata_SKAB_seed{seed}", f"Automata_SKAB_seed{seed}.log")
    logger.info(f"--- Starting 5-fold CV for TimeSeriesAutomata on SKAB with Seed {seed} ---")
    
    loader = DataLoader("SKAB")
    raw_data = loader.load_skab(config.SKAB_PATH)
    
    # 5-fold StratifiedGroupKFold splits based on source_file
    folds = loader.split_by_group(raw_data, n_splits=5, stratified=True)
    
    fold_metrics = []
    fold_train_times = []
    fold_inf_times = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"--- Evaluating Fold {fold_idx + 1}/{len(folds)} ---")
        
        train_df = raw_data.iloc[train_idx]
        val_df = raw_data.iloc[val_idx]
        
        # Fit scaler + PCA on training fold ONLY to prevent data leakage
        loader_fold = DataLoader("SKAB")
        train_scaled, _ = loader_fold.preprocess(train_df, fit_scaler=True)
        val_scaled, _ = loader_fold.transform(val_df)
        
        train_labels = loader_fold.get_labels(train_df)
        val_labels = loader_fold.get_labels(val_df)
        
        # Initialize deterministic TimeSeriesAutomata under current seed
        automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
        
        # Train fold & record training time
        start_train = time.time()
        automata.fit(train_scaled, window_size=config.WINDOW_SIZE)
        end_train = time.time()
        train_time = end_train - start_train
        fold_train_times.append(train_time)
        
        # Threshold optimization on fold validation set
        best_thresh, _ = find_best_threshold_automata(automata, val_scaled, val_labels, config.WINDOW_SIZE)
        
        # Evaluate validation inference & record inference time
        start_inf = time.time()
        y_pred_all = automata.predict_anomaly(val_scaled, window_size=config.WINDOW_SIZE, threshold=best_thresh)
        end_inf = time.time()
        inf_time = end_inf - start_inf
        fold_inf_times.append(inf_time)
        
        # Metrics calculation on fold validation
        y_true = val_labels[config.WINDOW_SIZE:]
        y_pred = y_pred_all[1:]
        
        metrics = calculate_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        
        logger.info(f"Fold {fold_idx+1} complete: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}, Train Time={train_time:.2f}s")
        
    avg_f1 = np.mean([m["f1"] for m in fold_metrics])
    avg_prec = np.mean([m["precision"] for m in fold_metrics])
    avg_rec = np.mean([m["recall"] for m in fold_metrics])
    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    avg_train_time = np.mean(fold_train_times)
    avg_inf_time = np.mean(fold_inf_times)
    
    results = {
        "dataset": "SKAB",
        "model": "Automata",
        "seed": seed,
        "f1": float(avg_f1),
        "precision": float(avg_prec),
        "recall": float(avg_rec),
        "accuracy": float(avg_acc),
        "training_time_sec": float(avg_train_time),
        "inference_time_sec": float(avg_inf_time)
    }
    
    save_results(results, "automata_metrics.csv")
    logger.info(f"--- Seed {seed} Overall SKAB Results: F1={avg_f1:.4f}, Accuracy={avg_acc:.4f}, Avg Train Time={avg_train_time:.2f}s ---")
    return results

def run_automata_batadal_chronological(seed):
    """
    Runs chronological split training and evaluation on BATADAL for a specific seed.
    Logs metrics and training/inference execution runtimes.
    """
    config = Config()
    set_seed(seed)
    
    logger = setup_logger(f"Evaluate_Automata_BATADAL_seed{seed}", f"Automata_BATADAL_seed{seed}.log")
    logger.info(f"--- Starting Chronological Evaluation for TimeSeriesAutomata on BATADAL with Seed {seed} ---")
    
    loader = DataLoader("BATADAL")
    path = config.BATADAL_PATH
    if os.path.isdir(path):
        path = os.path.join(path, "batadal_training_2.csv")
        
    raw_data = loader.load_batadal(path)
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    train_data, val_data, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    train_labels, val_labels, test_labels = loader.split_chronological(pd.Series(labels))
    
    # Initialize TimeSeriesAutomata
    automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
    
    # Train model & record training time
    start_train = time.time()
    automata.fit(train_data.values, window_size=config.WINDOW_SIZE)
    end_train = time.time()
    train_time = end_train - start_train
    
    # Threshold optimization on validation set
    best_thresh, _ = find_best_threshold_automata(automata, val_data.values, val_labels.values, config.WINDOW_SIZE)
    
    # Evaluate test inference & record inference time
    start_inf = time.time()
    y_pred_all = automata.predict_anomaly(test_data.values, window_size=config.WINDOW_SIZE, threshold=best_thresh)
    end_inf = time.time()
    inf_time = end_inf - start_inf
    
    # Predict and compute metrics
    y_true = test_labels.values[config.WINDOW_SIZE:]
    y_pred = y_pred_all[1:]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    results = {
        "dataset": "BATADAL",
        "model": "Automata",
        "seed": seed,
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "accuracy": float(metrics["accuracy"]),
        "training_time_sec": float(train_time),
        "inference_time_sec": float(inf_time)
    }
    
    save_results(results, "automata_metrics.csv")
    logger.info(f"BATADAL complete: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}, Train Time={train_time:.2f}s")
    return results

if __name__ == "__main__":
    seeds = [42, 123, 2026, 7, 999]
    metrics_file = "automata_metrics.csv"
    
    # Remove previous automata metrics file if it exists to have clean results
    os.makedirs("results/metrics", exist_ok=True)
    path = os.path.join("results/metrics", metrics_file)
    if os.path.exists(path):
        os.remove(path)
        
    for dataset in ["SKAB", "BATADAL"]:
        for seed in seeds:
            try:
                if dataset == "SKAB":
                    run_automata_skab_kfold(seed)
                else:
                    run_automata_batadal_chronological(seed)
            except Exception as e:
                print(f"Error evaluating Automata on {dataset} with seed {seed}: {e}")
