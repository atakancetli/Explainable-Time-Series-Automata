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
