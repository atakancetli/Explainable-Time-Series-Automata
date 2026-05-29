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

def evaluate_cross_dataset_automata(train_dataset, test_dataset, seed):
    """
    Evaluates TimeSeriesAutomata trained on train_dataset and tested on test_dataset.
    """
    config = Config()
    set_seed(seed)
    
    # 1. Load and preprocess training data
    loader_train = DataLoader(train_dataset)
    if train_dataset == "SKAB":
        raw_train = loader_train.load_skab(config.SKAB_PATH)
        scaled_train, _ = loader_train.preprocess(raw_train, fit_scaler=True)
    else:
        path = os.path.join(config.BATADAL_PATH, "batadal_training_2.csv")
        raw_train = loader_train.load_batadal(path)
        scaled_train, _ = loader_train.preprocess(raw_train, fit_scaler=True)
        # Use full BATADAL training split for robust training
        scaled_train_df = pd.DataFrame(scaled_train)
        scaled_train = scaled_train_df.iloc[:int(len(scaled_train)*config.TRAIN_RATIO)].values
        
    automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
    automata.fit(scaled_train, window_size=config.WINDOW_SIZE)
    
    # 2. Load and preprocess test data
    loader_test = DataLoader(test_dataset)
    if test_dataset == "SKAB":
        raw_test = loader_test.load_skab(config.SKAB_PATH)
        scaled_test_all, _ = loader_test.preprocess(raw_test, fit_scaler=True)
        # Use validation and test splits from GroupKFold or chronological
        test_labels_all = loader_test.get_labels(raw_test)
        train_len = int(len(scaled_test_all) * config.TRAIN_RATIO)
        val_len = int(len(scaled_test_all) * config.VAL_RATIO)
        
        val_data = scaled_test_all[train_len : train_len + val_len]
        val_labels = test_labels_all[train_len : train_len + val_len]
        test_data = scaled_test_all[train_len + val_len :]
        test_labels = test_labels_all[train_len + val_len :]
    else:
        path = os.path.join(config.BATADAL_PATH, "batadal_training_2.csv")
        raw_test = loader_test.load_batadal(path)
        scaled_test_all, _ = loader_test.preprocess(raw_test, fit_scaler=True)
        test_labels_all = loader_test.get_labels(raw_test)
        
        train_data_df, val_data_df, test_data_df = loader_test.split_chronological(pd.DataFrame(scaled_test_all))
        train_labels_df, val_labels_df, test_labels_df = loader_test.split_chronological(pd.Series(test_labels_all))
        
        val_data = val_data_df.values
        val_labels = val_labels_df.values
        test_data = test_data_df.values
        test_labels = test_labels_df.values
        
    # Optimize threshold on the test dataset's validation split
    from scripts.evaluate_automata import find_best_threshold_automata
    best_thresh, _ = find_best_threshold_automata(automata, val_data, val_labels, config.WINDOW_SIZE)
    
    # Predict anomalies on test split
    preds = automata.predict_anomaly(test_data, window_size=config.WINDOW_SIZE, threshold=best_thresh)
    y_true = test_labels[config.WINDOW_SIZE:]
    metrics = calculate_metrics(y_true, preds[1:])
    return metrics

def evaluate_cross_dataset_dl(model_type, train_dataset, test_dataset, seed):
    """
    Evaluates Deep Learning model trained on train_dataset and tested on test_dataset.
    """
    config = Config()
    set_seed(seed)
    
    # 1. Load and preprocess test data
    loader_test = DataLoader(test_dataset)
    if test_dataset == "SKAB":
        raw_test = loader_test.load_skab(config.SKAB_PATH)
        scaled_test_all, _ = loader_test.preprocess(raw_test, fit_scaler=True)
        test_labels_all = loader_test.get_labels(raw_test)
        
        # Chronological splits
        train_data, val_data, test_data = loader_test.split_chronological(pd.DataFrame(scaled_test_all))
        train_labels, val_labels, test_labels = loader_test.split_chronological(pd.Series(test_labels_all))
    else:
        path = os.path.join(config.BATADAL_PATH, "batadal_training_2.csv")
        raw_test = loader_test.load_batadal(path)
        scaled_test_all, _ = loader_test.preprocess(raw_test, fit_scaler=True)
        test_labels_all = loader_test.get_labels(raw_test)
        
        train_data, val_data, test_data = loader_test.split_chronological(pd.DataFrame(scaled_test_all))
        train_labels, val_labels, test_labels = loader_test.split_chronological(pd.Series(test_labels_all))
        
    _, val_loader, test_loader = loader_test.get_dataloaders(
        train_data.values, val_data.values, test_data.values,
        train_labels.values, val_labels.values, test_labels.values
    )
    
    # 2. Initialize and load model
    model = ModelFactory.get_model(
        model_type, 
        scaled_test_all.shape[1], 
        config.HIDDEN_SIZE, 
        config.NUM_LAYERS,
        dropout=0.2,
        window_size=config.WINDOW_SIZE
    ).to(config.DEVICE)
    
    if train_dataset == "SKAB":
        checkpoint_path = f"checkpoints/SKAB_{model_type}_seed{seed}_fold1_best.pth"
    else:
        checkpoint_path = f"checkpoints/BATADAL_{model_type}_seed{seed}_best.pth"
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    model = load_model(model, checkpoint_path, config.DEVICE)
    model.eval()
    
    # Find best threshold on the test dataset's validation split (clean)
    from scripts.train_dl import find_best_threshold
    best_thresh, _ = find_best_threshold(model, val_loader, val_labels.values, config.DEVICE)
    
    # Predict anomalies on test split
    y_probs = predict(model, test_loader, config.DEVICE)
    window_size = len(test_labels) - len(y_probs)
    y_true = test_labels.values[window_size:]
    y_pred = (y_probs >= best_thresh).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred)
    return metrics

def evaluate_automata_window_sensitivity(window_size, seed, dataset_name="BATADAL"):
    """
    Wrapper for window size parameter sensitivity analysis.
    """
    return evaluate_automata_sensitivity("window_size", window_size, seed, dataset_name)

def evaluate_automata_alphabet_sensitivity(alphabet_size, seed, dataset_name="BATADAL"):
    """
    Wrapper for alphabet size parameter sensitivity analysis.
    """
    return evaluate_automata_sensitivity("alphabet_size", alphabet_size, seed, dataset_name)

def evaluate_automata_sensitivity(param_name, param_value, seed, dataset_name="BATADAL"):

    """
    Fits and evaluates TimeSeriesAutomata on dataset_name with param_name set to param_value.
    Holding other parameters at standard defaults (window_size=10, alphabet_size=5).
    """
    config = Config()
    set_seed(seed)
    
    # Configure custom parameters
    w_size = param_value if param_name == "window_size" else 10
    a_size = param_value if param_name == "alphabet_size" else 5
    
    loader = DataLoader(dataset_name)
    if dataset_name == "SKAB":
        raw_data = loader.load_skab(config.SKAB_PATH)
        scaled_all, _ = loader.preprocess(raw_data, fit_scaler=True)
        labels_all = loader.get_labels(raw_data)
        
        train_len = int(len(scaled_all) * config.TRAIN_RATIO)
        val_len = int(len(scaled_all) * config.VAL_RATIO)
        
        train_data = scaled_all[:train_len]
        val_data = scaled_all[train_len : train_len + val_len]
        val_labels = labels_all[train_len : train_len + val_len]
        test_data = scaled_all[train_len + val_len :]
        test_labels = labels_all[train_len + val_len :]
    else:
        path = os.path.join(config.BATADAL_PATH, "batadal_training_2.csv")
        raw_data = loader.load_batadal(path)
        scaled_all, _ = loader.preprocess(raw_data, fit_scaler=True)
        labels_all = loader.get_labels(raw_data)
        
        train_data_df, val_data_df, test_data_df = loader.split_chronological(pd.DataFrame(scaled_all))
        train_labels_df, val_labels_df, test_labels_df = loader.split_chronological(pd.Series(labels_all))
        
        train_data = train_data_df.values
        val_data = val_data_df.values
        val_labels = val_labels_df.values
        test_data = test_data_df.values
        test_labels = test_labels_df.values
        
    automata = TimeSeriesAutomata(alphabet_size=a_size, word_size=4)
    automata.fit(train_data, window_size=w_size)
    
    # Optimize threshold on the validation split
    from scripts.evaluate_automata import find_best_threshold_automata
    best_thresh, _ = find_best_threshold_automata(automata, val_data, val_labels, w_size)
    
    # Predict anomalies on test split
    preds = automata.predict_anomaly(test_data, window_size=w_size, threshold=best_thresh)
    y_true = test_labels[w_size:]
    metrics = calculate_metrics(y_true, preds[1:])
    return metrics

def evaluate_dl_skab_kfold_robustness(model_type, seed, noise_scale=0.1):



    """
    Evaluates Deep Learning models on SKAB with 5-fold cross-validation and noise injection.
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
        loader_fold.fit(train_df)
        val_scaled, _ = loader_fold.transform(val_df)
        val_labels = loader_fold.get_labels(val_df)
        
        # Clean validation loader for threshold tuning & original metrics
        _, val_loader_clean = loader_fold.get_fold_dataloaders(
            val_scaled, val_scaled, val_labels, val_labels
        )
        
        # Initialize model
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
        
        # Find best threshold on clean validation data
        from scripts.train_dl import find_best_threshold
        best_thresh, _ = find_best_threshold(model, val_loader_clean, val_labels, config.DEVICE)
        
        # Clean predictions
        y_probs_clean = predict(model, val_loader_clean, config.DEVICE)
        window_size = len(val_labels) - len(y_probs_clean)
        y_true = val_labels[window_size:]
        y_pred_clean = (y_probs_clean >= best_thresh).astype(int)
        fold_metrics_clean.append(calculate_metrics(y_true, y_pred_clean))
        
        # Noisy predictions
        noisy_val_scaled = inject_gaussian_noise(val_scaled, scale=noise_scale)
        _, val_loader_noisy = loader_fold.get_fold_dataloaders(
            noisy_val_scaled, noisy_val_scaled, val_labels, val_labels
        )
        y_probs_noisy = predict(model, val_loader_noisy, config.DEVICE)
        y_pred_noisy = (y_probs_noisy >= best_thresh).astype(int)
        fold_metrics_noisy.append(calculate_metrics(y_true, y_pred_noisy))
        
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

def evaluate_dl_batadal_robustness(model_type, seed, noise_scale=0.1):
    """
    Evaluates Deep Learning models on BATADAL chronological split with noise injection.
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
    
    # Initialize clean validation and test loaders
    _, val_loader, test_loader_clean = loader.get_dataloaders(
        train_data.values, val_data.values, test_data.values,
        train_labels.values, val_labels.values, test_labels.values
    )
    
    # Initialize model
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
    
    # Find best threshold on clean validation data
    from scripts.train_dl import find_best_threshold
    best_thresh, _ = find_best_threshold(model, val_loader, val_labels.values, config.DEVICE)
    
    # Original evaluation
    y_probs_clean = predict(model, test_loader_clean, config.DEVICE)
    window_size = len(test_labels) - len(y_probs_clean)
    y_true = test_labels.values[window_size:]
    y_pred_clean = (y_probs_clean >= best_thresh).astype(int)
    metrics_clean = calculate_metrics(y_true, y_pred_clean)
    
    # Noisy evaluation
    noisy_test_data = inject_gaussian_noise(test_data.values, scale=noise_scale)
    _, _, test_loader_noisy = loader.get_dataloaders(
        train_data.values, val_data.values, noisy_test_data,
        train_labels.values, val_labels.values, test_labels.values
    )
    y_probs_noisy = predict(model, test_loader_noisy, config.DEVICE)
    y_pred_noisy = (y_probs_noisy >= best_thresh).astype(int)
    metrics_noisy = calculate_metrics(y_true, y_pred_noisy)
    
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

def run_cross_dataset_sweeps():
    """
    Orchestrates cross-dataset validation experiments.
    """
    seeds = [42, 123, 2026, 7, 999]
    metrics_file = "cross_dataset_results.csv"
    os.makedirs("results/metrics", exist_ok=True)
    path = os.path.join("results/metrics", metrics_file)
    if os.path.exists(path):
        os.remove(path)
        
    all_results = []
    
    # We want cross-dataset: SKAB->BATADAL and BATADAL->SKAB
    dataset_pairs = [
        ("SKAB", "BATADAL"),
        ("BATADAL", "SKAB")
    ]
    
    for train_ds, test_ds in dataset_pairs:
        for model in ["Automata", "LSTM", "GRU", "CNN"]:
            print(f"Sweeping cross-dataset for Train: {train_ds} -> Test: {test_ds} | Model: {model}...")
            run_metrics = []
            for seed in seeds:
                try:
                    if model == "Automata":
                        metrics = evaluate_cross_dataset_automata(train_ds, test_ds, seed)
                    else:
                        metrics = evaluate_cross_dataset_dl(model, train_ds, test_ds, seed)
                    run_metrics.append(metrics)
                except Exception as e:
                    print(f"Error in cross-dataset evaluation of {model} from {train_ds} to {test_ds} with seed {seed}: {e}")
                    
            if len(run_metrics) > 0:
                avg_metrics = {
                    "train_dataset": train_ds,
                    "test_dataset": test_ds,
                    "model": model,
                    "f1": float(np.mean([m["f1"] for m in run_metrics])),
                    "precision": float(np.mean([m["precision"] for m in run_metrics])),
                    "recall": float(np.mean([m["recall"] for m in run_metrics])),
                    "accuracy": float(np.mean([m["accuracy"] for m in run_metrics]))
                }
                save_results(avg_metrics, metrics_file)
                all_results.append(avg_metrics)
                print(f"==> Avg Cross-Dataset Result for {train_ds} -> {test_ds} | {model}: F1 = {avg_metrics['f1']:.4f} | Accuracy = {avg_metrics['accuracy']:.4f}")
                
    return all_results

def compile_academic_tables():
    """
    Compiles and prints the academic markdown tables for the report.
    """
    print("\n" + "="*80)
    print("ACADEMIC EXPERIMENT RESULTS REPORT COMPILATION")
    print("="*80)
    
    # 1. Table 2: Robustness Anomaly Detection (averaging across noise scales)
    robustness_path = "results/metrics/robustness_sweep_results.csv"
    if os.path.exists(robustness_path):
        df_rob = pd.read_csv(robustness_path)
        print("\n### Tablo 2: Gürültü Etkisi Analizi (Ortalama F1-Score)")
        print("| Model | Veri Seti | Orijinal F1 | Gürültü F1 (0.05) | Gürültü F1 (0.1) | Gürültü F1 (0.15) | Gürültü F1 (0.2) | Gürültü F1 (0.25) |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        
        for dataset in ["SKAB", "BATADAL"]:
            for model in ["Automata", "LSTM", "GRU", "CNN"]:
                sub = df_rob[(df_rob['dataset'] == dataset) & (df_rob['model'] == model)]
                if not sub.empty:
                    # Get baseline from noise scale 0.05's orig_f1
                    orig_f1 = sub.iloc[0]['orig_f1']
                    f1s = []
                    for scale in [0.05, 0.1, 0.15, 0.2, 0.25]:
                        row = sub[np.isclose(sub['noise_scale'], scale)]
                        if not row.empty:
                            f1s.append(f"{row.iloc[0]['noisy_f1']:.4f}")
                        else:
                            f1s.append("-")
                    print(f"| {model} | {dataset} | {orig_f1:.4f} | " + " | ".join(f1s) + " |")
                    
    # 2. Table 3: Cross-Dataset Generalizability Matrix
    cross_path = "results/metrics/cross_dataset_results.csv"
    if os.path.exists(cross_path):
        df_cross = pd.read_csv(cross_path)
        print("\n### Tablo 3: Cross-Dataset Performans Karşılaştırması (F1-Score)")
        
        # We need a cross matrix for each model
        for model in ["Automata", "LSTM", "GRU", "CNN"]:
            print(f"\n**Model: {model}**")
            print("| Train \\ Test | SKAB | BATADAL |")
            print("| --- | --- | --- |")
            
            # Row 1: Train: SKAB
            skab_skab_val = "-"
            # Load in-domain metrics if they exist in robustness
            if os.path.exists(robustness_path):
                df_rob = pd.read_csv(robustness_path)
                sub_id = df_rob[(df_rob['dataset'] == 'SKAB') & (df_rob['model'] == model)]
                if not sub_id.empty:
                    skab_skab_val = f"{sub_id.iloc[0]['orig_f1']:.4f}"
            skab_batadal_val = "-"
            row_sb = df_cross[(df_cross['train_dataset'] == 'SKAB') & (df_cross['test_dataset'] == 'BATADAL') & (df_cross['model'] == model)]
            if not row_sb.empty:
                skab_batadal_val = f"{row_sb.iloc[0]['f1']:.4f}"
            print(f"| Train: SKAB | {skab_skab_val} | {skab_batadal_val} |")
            
            # Row 2: Train: BATADAL
            batadal_skab_val = "-"
            row_bs = df_cross[(df_cross['train_dataset'] == 'BATADAL') & (df_cross['test_dataset'] == 'SKAB') & (df_cross['model'] == model)]
            if not row_bs.empty:
                batadal_skab_val = f"{row_bs.iloc[0]['f1']:.4f}"
            batadal_batadal_val = "-"
            if os.path.exists(robustness_path):
                sub_id2 = df_rob[(df_rob['dataset'] == 'BATADAL') & (df_rob['model'] == model)]
                if not sub_id2.empty:
                    batadal_batadal_val = f"{sub_id2.iloc[0]['orig_f1']:.4f}"
            print(f"| Train: BATADAL | {batadal_skab_val} | {batadal_batadal_val} |")
            
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Running Robustness Sweeps...")
    run_multi_model_robustness_sweeps()
    print("\nRunning Cross-Dataset Sweeps...")
    run_cross_dataset_sweeps()
    print("\nCompiling Academic Tables...")
    compile_academic_tables()


