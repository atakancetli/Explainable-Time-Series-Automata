import os
import torch
import numpy as np
import pandas as pd
from configs.config import Config
from models.model_factory import ModelFactory
from models.automata import TimeSeriesAutomata
from utils.data_loader import DataLoader, TimeSeriesDataset
from utils.train_utils import load_model, predict
from utils.metrics import calculate_metrics, save_results
from utils.robustness import inject_gaussian_noise, calculate_unseen_metrics
from torch.utils.data import DataLoader as TorchDataLoader

def evaluate_model_robustness(dataset_name, model_type, noise_scale=0.1):
    config = Config()
    loader = DataLoader(dataset_name)
    
    # Load dataset
    if dataset_name == "SKAB":
        raw_data = loader.load_skab(config.SKAB_PATH)
    else:
        path = config.BATADAL_PATH
        if os.path.isdir(path):
            path = os.path.join(path, "batadal_training_2.csv")
        raw_data = loader.load_batadal(path)
        
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    # Chronological splitting
    train_data, val_data, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    train_labels, val_labels, test_labels = loader.split_chronological(pd.Series(labels))
    
    # Ground truth aligned test labels
    y_true = test_labels.values[config.WINDOW_SIZE:]
    
    if model_type == "Automata":
        # 1. Automata Model fitting and evaluation
        automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
        automata.fit(train_data.values, window_size=config.WINDOW_SIZE)
        
        # Original evaluation
        preds_orig = automata.predict_anomaly(test_data.values, window_size=config.WINDOW_SIZE, threshold=0.01)
        # S0 has transition count 0, so predictions starts at index 1 to align with test_labels[WINDOW_SIZE:]
        metrics_orig = calculate_metrics(y_true, preds_orig[1:])
        
        # Noisy evaluation
        noisy_test = inject_gaussian_noise(test_data.values, scale=noise_scale)
        preds_noisy = automata.predict_anomaly(noisy_test, window_size=config.WINDOW_SIZE, threshold=0.01)
        metrics_noisy = calculate_metrics(y_true, preds_noisy[1:])
        
        # Unseen metrics
        unseen = calculate_unseen_metrics(
            automata, train_data.values, test_data.values, train_labels.values, test_labels.values,
            window_size=config.WINDOW_SIZE, threshold=0.01
        )
        
        results = {
            "dataset": dataset_name,
            "model": "Automata",
            "noise_scale": noise_scale,
            "orig_f1": metrics_orig["f1"],
            "orig_precision": metrics_orig["precision"],
            "orig_recall": metrics_orig["recall"],
            "noisy_f1": metrics_noisy["f1"],
            "noisy_precision": metrics_noisy["precision"],
            "noisy_recall": metrics_noisy["recall"],
            "unseen_detection_rate": unseen["detection_rate"],
            "unseen_mapping_accuracy": unseen["mapping_accuracy"],
            "unseen_count": unseen["unseen_count"]
        }
    else:
        # 2. Deep Learning models loading and evaluation
        model = ModelFactory.get_model(
            model_type, 
            scaled_data.shape[1], 
            config.HIDDEN_SIZE, 
            config.NUM_LAYERS,
            dropout=0.2,
            window_size=config.WINDOW_SIZE
        ).to(config.DEVICE)
        
        checkpoint_path = f"checkpoints/{dataset_name}_{model_type}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {model_type} on {dataset_name}: Checkpoint not found at {checkpoint_path}")
            return None
            
        model = load_model(model, checkpoint_path, config.DEVICE)
        
        # Original loader & metrics
        test_ds = TimeSeriesDataset(test_data.values, config.WINDOW_SIZE, test_labels.values)
        test_loader = TorchDataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        preds_orig_probs = predict(model, test_loader, config.DEVICE)
        preds_orig = (preds_orig_probs > 0.5).astype(int)
        metrics_orig = calculate_metrics(y_true, preds_orig)
        
        # Noisy loader & metrics
        noisy_test_feats = inject_gaussian_noise(test_data.values, scale=noise_scale)
        noisy_test_ds = TimeSeriesDataset(noisy_test_feats, config.WINDOW_SIZE, test_labels.values)
        noisy_loader = TorchDataLoader(noisy_test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        preds_noisy_probs = predict(model, noisy_loader, config.DEVICE)
        preds_noisy = (preds_noisy_probs > 0.5).astype(int)
        metrics_noisy = calculate_metrics(y_true, preds_noisy)
        
        # DL models do not have Levenshtein mapping, so default to 0.0
        results = {
            "dataset": dataset_name,
            "model": model_type,
            "noise_scale": noise_scale,
            "orig_f1": metrics_orig["f1"],
            "orig_precision": metrics_orig["precision"],
            "orig_recall": metrics_orig["recall"],
            "noisy_f1": metrics_noisy["f1"],
            "noisy_precision": metrics_noisy["precision"],
            "noisy_recall": metrics_noisy["recall"],
            "unseen_detection_rate": 0.0,
            "unseen_mapping_accuracy": 0.0,
            "unseen_count": 0
        }
        
    save_results(results, "robustness_metrics.csv")
    print(f"[{dataset_name} | {model_type}] Scale={noise_scale} | Orig F1={results['orig_f1']:.4f} --> Noisy F1={results['noisy_f1']:.4f}")
    return results

if __name__ == "__main__":
    print("==========================================================")
    print("Running Multi-Model Robustness Testing Suite")
    print("==========================================================")
    for dataset in ["SKAB", "BATADAL"]:
        for model in ["Automata", "LSTM", "GRU", "CNN"]:
            for scale in [0.1, 0.2]:
                try:
                    evaluate_model_robustness(dataset, model, noise_scale=scale)
                except Exception as e:
                    print(f"Error evaluating {model} on {dataset} with scale {scale}: {e}")
