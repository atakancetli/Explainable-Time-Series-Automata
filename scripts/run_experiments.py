import os
import torch
import numpy as np
import pandas as pd
import time
from configs.config import Config
from models.model_factory import ModelFactory
from models.automata import TimeSeriesAutomata
from utils.data_loader import DataLoader, TimeSeriesDataset
from utils.train_utils import load_model, predict, set_seed
from utils.metrics import calculate_metrics, save_results
from utils.robustness import inject_gaussian_noise, calculate_unseen_metrics
from torch.utils.data import DataLoader as TorchDataLoader

def evaluate_automata_skab_kfold_robustness(seed, noise_scale=0.1):
    """
    Evaluates Automata model on SKAB 5-fold CV with noise perturbation on test/validation sets.
    """
    config = Config()
    set_seed(seed)
    
    loader = DataLoader("SKAB")
    raw_data = loader.load_skab(config.SKAB_PATH)
    
    folds = loader.split_by_group(raw_data, n_splits=5, stratified=True)
    
    fold_metrics_clean = []
    fold_metrics_noisy = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_df = raw_data.iloc[train_idx]
        val_df = raw_data.iloc[val_idx]
        
        loader_fold = DataLoader("SKAB")
        train_scaled, _ = loader_fold.preprocess(train_df, fit_scaler=True)
        val_scaled, _ = loader_fold.transform(val_df)
        
        train_labels = loader_fold.get_labels(train_df)
        val_labels = loader_fold.get_labels(val_df)
        
        automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
        automata.fit(train_scaled, window_size=config.WINDOW_SIZE)
        
        # Optimize threshold on clean validation data
        from scripts.evaluate_automata import find_best_threshold_automata
        best_thresh, _ = find_best_threshold_automata(automata, val_scaled, val_labels, config.WINDOW_SIZE)
        
        # Original evaluation
        preds_clean_all = automata.predict_anomaly(val_scaled, window_size=config.WINDOW_SIZE, threshold=best_thresh)
        y_true = val_labels[config.WINDOW_SIZE:]
        fold_metrics_clean.append(calculate_metrics(y_true, preds_clean_all[1:]))
        
        # Noisy evaluation
        noisy_val = inject_gaussian_noise(val_scaled, scale=noise_scale)
        preds_noisy_all = automata.predict_anomaly(noisy_val, window_size=config.WINDOW_SIZE, threshold=best_thresh)
        fold_metrics_noisy.append(calculate_metrics(y_true, preds_noisy_all[1:]))
        
    avg_f1_clean = np.mean([m["f1"] for m in fold_metrics_clean])
    avg_prec_clean = np.mean([m["precision"] for m in fold_metrics_clean])
    avg_rec_clean = np.mean([m["recall"] for m in fold_metrics_clean])
    avg_acc_clean = np.mean([m["accuracy"] for m in fold_metrics_clean])
    
    avg_f1_noisy = np.mean([m["f1"] for m in fold_metrics_noisy])
    avg_prec_noisy = np.mean([m["precision"] for m in fold_metrics_noisy])
    avg_rec_noisy = np.mean([m["recall"] for m in fold_metrics_noisy])
    avg_acc_noisy = np.mean([m["accuracy"] for m in fold_metrics_noisy])
    
    return {
        "orig_f1": float(avg_f1_clean),
        "orig_precision": float(avg_prec_clean),
        "orig_recall": float(avg_rec_clean),
        "orig_accuracy": float(avg_acc_clean),
        "noisy_f1": float(avg_f1_noisy),
        "noisy_precision": float(avg_prec_noisy),
        "noisy_recall": float(avg_rec_noisy),
        "noisy_accuracy": float(avg_acc_noisy)
    }

def evaluate_automata_batadal_robustness(seed, noise_scale=0.1):
    """
    Evaluates Automata model on BATADAL chronological splits with noise perturbation.
    """
    config = Config()
    set_seed(seed)
    
    loader = DataLoader("BATADAL")
    path = config.BATADAL_PATH
    if os.path.isdir(path):
        path = os.path.join(path, "batadal_training_2.csv")
    raw_data = loader.load_batadal(path)
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    train_data, val_data, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    train_labels, val_labels, test_labels = loader.split_chronological(pd.Series(labels))
    
    automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
    automata.fit(train_data.values, window_size=config.WINDOW_SIZE)
    
    from scripts.evaluate_automata import find_best_threshold_automata
    best_thresh, _ = find_best_threshold_automata(automata, val_data.values, val_labels.values, config.WINDOW_SIZE)
    
    # Original evaluation
    preds_clean_all = automata.predict_anomaly(test_data.values, window_size=config.WINDOW_SIZE, threshold=best_thresh)
    y_true = test_labels.values[config.WINDOW_SIZE:]
    metrics_clean = calculate_metrics(y_true, preds_clean_all[1:])
    
    # Noisy evaluation
    noisy_test = inject_gaussian_noise(test_data.values, scale=noise_scale)
    preds_noisy_all = automata.predict_anomaly(noisy_test, window_size=config.WINDOW_SIZE, threshold=best_thresh)
    metrics_noisy = calculate_metrics(y_true, preds_noisy_all[1:])
    
    return {
        "orig_f1": float(metrics_clean["f1"]),
        "orig_precision": float(metrics_clean["precision"]),
        "orig_recall": float(metrics_clean["recall"]),
        "orig_accuracy": float(metrics_clean["accuracy"]),
        "noisy_f1": float(metrics_noisy["f1"]),
        "noisy_precision": float(metrics_noisy["precision"]),
        "noisy_recall": float(metrics_noisy["recall"]),
        "noisy_accuracy": float(metrics_noisy["accuracy"])
    }

def evaluate_dl_skab_kfold_robustness(model_type, seed, noise_scale=0.1):
    """
    Placeholder for Deep Learning models on SKAB with GroupKFold splits and noise.
    """
    return {
        "orig_f1": 0.0, "orig_precision": 0.0, "orig_recall": 0.0, "orig_accuracy": 0.0,
        "noisy_f1": 0.0, "noisy_precision": 0.0, "noisy_recall": 0.0, "noisy_accuracy": 0.0
    }

def evaluate_dl_batadal_robustness(model_type, seed, noise_scale=0.1):
    """
    Placeholder for Deep Learning models on BATADAL chronological splits with noise.
    """
    return {
        "orig_f1": 0.0, "orig_precision": 0.0, "orig_recall": 0.0, "orig_accuracy": 0.0,
        "noisy_f1": 0.0, "noisy_precision": 0.0, "noisy_recall": 0.0, "noisy_accuracy": 0.0
    }

def run_multi_model_robustness_sweeps(noise_scales=[0.05, 0.1, 0.15, 0.2, 0.25]):
    """
    Coordinates multi-model and multi-dataset robustness evaluations.
    """
    seeds = [42, 123, 2026, 7, 999]
    metrics_file = "robustness_sweep_results.csv"
    os.makedirs("results/metrics", exist_ok=True)
    path = os.path.join("results/metrics", metrics_file)
    if os.path.exists(path):
        os.remove(path)
        
    all_results = []
    
    for dataset in ["SKAB", "BATADAL"]:
        for model in ["Automata", "LSTM", "GRU", "CNN"]:
            for scale in noise_scales:
                print(f"Sweeping robustness for {dataset} | {model} | Noise={scale}...")
                scale_results = []
                for seed in seeds:
                    try:
                        if dataset == "SKAB":
                            if model == "Automata":
                                metrics = evaluate_automata_skab_kfold_robustness(seed, scale)
                            else:
                                metrics = evaluate_dl_skab_kfold_robustness(model, seed, scale)
                        else:
                            if model == "Automata":
                                metrics = evaluate_automata_batadal_robustness(seed, scale)
                            else:
                                metrics = evaluate_dl_batadal_robustness(model, seed, scale)
                        
                        scale_results.append(metrics)
                    except Exception as e:
                        print(f"Error evaluating {model} on {dataset} with seed {seed}: {e}")
                
                if len(scale_results) > 0:
                    avg_metrics = {
                        "dataset": dataset,
                        "model": model,
                        "noise_scale": scale,
                        "orig_f1": float(np.mean([r["orig_f1"] for r in scale_results])),
                        "orig_precision": float(np.mean([r["orig_precision"] for r in scale_results])),
                        "orig_recall": float(np.mean([r["orig_recall"] for r in scale_results])),
                        "orig_accuracy": float(np.mean([r["orig_accuracy"] for r in scale_results])),
                        "noisy_f1": float(np.mean([r["noisy_f1"] for r in scale_results])),
                        "noisy_precision": float(np.mean([r["noisy_precision"] for r in scale_results])),
                        "noisy_recall": float(np.mean([r["noisy_recall"] for r in scale_results])),
                        "noisy_accuracy": float(np.mean([r["noisy_accuracy"] for r in scale_results]))
                    }
                    save_results(avg_metrics, metrics_file)
                    all_results.append(avg_metrics)
                    print(f"==> Avg Results for {dataset} {model} (Noise {scale}): Clean F1 = {avg_metrics['orig_f1']:.4f} | Noisy F1 = {avg_metrics['noisy_f1']:.4f}")
    
    return all_results

if __name__ == "__main__":
    run_multi_model_robustness_sweeps()
