import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from configs.config import Config
from models.model_factory import ModelFactory
from models.automata import TimeSeriesAutomata
from utils.data_loader import DataLoader
from utils.train_utils import load_model, predict, set_seed

def plot_confusion_matrices(model_preds_dict, y_true_dict, save_path):
    """
    Plots a side-by-side grid of confusion matrices for each model in a premium academic style.
    """
    models = list(model_preds_dict.keys())
    num_models = len(models)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="white")
    
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))
    if num_models == 1:
        axes = [axes]
        
    for idx, model_name in enumerate(models):
        y_true = np.array(y_true_dict[model_name]).astype(int)
        y_pred = np.array(model_preds_dict[model_name]).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot styled heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            ax=axes[idx],
            cbar=False,
            square=True,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"]
        )
        
        # Add titles and labels
        axes[idx].set_title(f"{model_name} Confusion Matrix", fontsize=14, pad=10, weight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=12, labelpad=8)
        if idx == 0:
            axes[idx].set_ylabel("True Label", fontsize=12, labelpad=8)
        else:
            axes[idx].set_ylabel("")
            
        # Draw explicit borders
        for _, spine in axes[idx].spines.items():
            spine.set_visible(True)
            spine.set_color("#cccccc")
            spine.set_linewidth(1.5)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"==> Saved confusion matrices grid to: {save_path}")

def plot_roc_pr_curves(model_scores_dict, y_true_dict, save_path_prefix):
    """
    Plots side-by-side ROC and Precision-Recall curves in standard academic format.
    """
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    sns.set_theme(style="ticks")
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    colors = {
        "Automata": "#1f77b4", # Sleek Blue
        "LSTM": "#ff7f0e",     # Sleek Orange
        "GRU": "#2ca02c",      # Sleek Green
        "CNN": "#d62728"       # Sleek Red
    }
    
    # 1. ROC Curves
    ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="#888888", label="Random Baseline")
    
    for model_name, scores in model_scores_dict.items():
        y_true = np.array(y_true_dict[model_name]).astype(int)
        y_score = np.array(scores)
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        color = colors.get(model_name, "#777777")
        ax_roc.plot(fpr, tpr, color=color, lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")
        
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=12, labelpad=8)
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=12, labelpad=8)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, weight="bold", pad=12)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=10)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    
    # 2. Precision-Recall Curves
    for model_name, scores in model_scores_dict.items():
        y_true = np.array(y_true_dict[model_name]).astype(int)
        y_score = np.array(scores)
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        
        color = colors.get(model_name, "#777777")
        ax_pr.plot(recall, precision, color=color, lw=2.5, label=f"{model_name} (AP = {avg_precision:.4f})")
        
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall", fontsize=12, labelpad=8)
    ax_pr.set_ylabel("Precision", fontsize=12, labelpad=8)
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=13, weight="bold", pad=12)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=10)
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    roc_save_path = f"{save_path_prefix}_roc_pr_curves.png"
    plt.savefig(roc_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"==> Saved ROC and Precision-Recall curves to: {roc_save_path}")

def plot_sensitivity_heatmaps(save_path):
    """
    Generates a double-panel heatmap representing the F1-score parameter sensitivity
    matrix across window size and alphabet size for both SKAB and BATADAL datasets.
    """
    csv_path = "results/metrics/sensitivity_results.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: sensitivity results not found at: {csv_path}. Skipping heatmap generation.")
        return
        
    df = pd.read_csv(csv_path)
    datasets = ["SKAB", "BATADAL"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.set_theme(style="white")
    
    for idx, dataset in enumerate(datasets):
        sub_df = df[df["dataset"] == dataset]
        if sub_df.empty:
            continue
            
        # Construct matrix manually to ensure clean formatting
        matrix_data = []
        row_names = ["Pencere Boyutu (w)", "Alfabe Boyutu (a)"]
        param_names = ["window_size", "alphabet_size"]
        col_names = ["Değer = 3", "Değer = 4", "Değer = 5", "Değer = 6"]
        
        for param in param_names:
            row_vals = []
            for val in [3, 4, 5, 6]:
                row = sub_df[(sub_df["parameter"] == param) & (sub_df["value"] == val)]
                if not row.empty:
                    row_vals.append(row.iloc[0]["f1"])
                else:
                    row_vals.append(0.0)
            matrix_data.append(row_vals)
            
        heatmap_df = pd.DataFrame(matrix_data, index=row_names, columns=col_names)
        
        # Draw the heatmap
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            ax=axes[idx],
            cbar=True,
            square=False,
            annot_kws={"size": 12, "weight": "bold"},
            linewidths=1.5,
            linecolor="#ffffff"
        )
        
        axes[idx].set_title(f"{dataset} Parametre Duyarlılık Analizi", fontsize=13, weight="bold", pad=12)
        # Stylize labels
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0, fontsize=11, weight="bold")
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), fontsize=11)
        
        # Outline boundaries
        for _, spine in axes[idx].spines.items():
            spine.set_visible(True)
            spine.set_color("#cccccc")
            spine.set_linewidth(1.5)
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"==> Saved parameter sensitivity heatmaps to: {save_path}")

def plot_automata_transitions(save_path):
    """
    Fits TimeSeriesAutomata on BATADAL training dataset and plots the transition
    probability matrix of the top 10 most visited states as an annotated heatmap.
    """
    config = Config()
    set_seed(42)
    
    loader = DataLoader("BATADAL")
    path = config.BATADAL_PATH
    if os.path.isdir(path):
        path = os.path.join(path, "batadal_training_2.csv")
    raw_data = loader.load_batadal(path)
    scaled_data, _ = loader.preprocess(raw_data)
    
    # Train data split
    train_data_df, _, _ = loader.split_chronological(pd.DataFrame(scaled_data))
    train_data = train_data_df.values
    
    # Fit Automata
    automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
    automata.fit(train_data, window_size=config.WINDOW_SIZE)
    
    # Generate states in training data to count frequency
    states = automata.generate_states(train_data, window_size=config.WINDOW_SIZE)
    
    # Count frequencies of each state
    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1
        
    # Get top 10 most frequent states
    top_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_state_names = [state[0] for state in top_states]
    
    if len(top_state_names) < 2:
        print("Warning: not enough states found in Automata to plot transition matrix.")
        return
        
    # Construct transition matrix for top states
    n_states = len(top_state_names)
    matrix = np.zeros((n_states, n_states))
    
    for i, s_from in enumerate(top_state_names):
        if hasattr(automata, "transitions") and s_from in automata.transitions:
            outgoing = automata.transitions[s_from]
            total_outgoing = sum(outgoing.values())
            for j, s_to in enumerate(top_state_names):
                if total_outgoing > 0:
                    matrix[i][j] = outgoing.get(s_to, 0) / total_outgoing
            
    # Plot transition heatmap
    plt.figure(figsize=(9, 7.5))
    sns.set_theme(style="white")
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="Oranges",
        xticklabels=top_state_names,
        yticklabels=top_state_names,
        cbar=True,
        square=True,
        annot_kws={"size": 10, "weight": "bold"},
        linewidths=1.0,
        linecolor="#ffffff"
    )
    
    plt.title("TimeSeriesAutomata Durum Geçiş Olasılıkları Matrisi\n(En Sık Ziyaret Edilen 10 Durum)", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Hedef Durum ($S_t$)", fontsize=11, labelpad=8)
    plt.ylabel("Kaynak Durum ($S_{t-1}$)", fontsize=11, labelpad=8)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Draw frame
    ax = plt.gca()
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#cccccc")
        spine.set_linewidth(1.5)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"==> Saved symbolic automata transition matrix heatmap to: {save_path}")

def get_model_scores(model_type, seed, dataset_name):
    """
    Returns (continuous anomaly scores, labels) for ROC/PR curve plotting.
    Ensures correct scaling, data loaders, PCA, folds, and checkpoint loading.
    """
    config = Config()
    set_seed(seed)
    
    all_scores = []
    all_labels = []
    
    if dataset_name == "SKAB":
        loader = DataLoader("SKAB")
        raw_data = loader.load_skab(config.SKAB_PATH)
        folds = loader.split_by_group(raw_data, n_splits=5, stratified=True)
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_df = raw_data.iloc[train_idx]
            val_df = raw_data.iloc[val_idx]
            
            loader_fold = DataLoader("SKAB")
            train_scaled, _ = loader_fold.preprocess(train_df, fit_scaler=True)
            val_scaled, _ = loader_fold.transform(val_df)
            
            train_labels = loader_fold.get_labels(train_df)
            val_labels = loader_fold.get_labels(val_df)
            
            if model_type == "Automata":
                automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
                automata.fit(train_scaled, window_size=config.WINDOW_SIZE)
                
                states = automata.generate_states(val_scaled, window_size=config.WINDOW_SIZE)
                num_windows = len(states)
                scores = np.zeros(num_windows)
                for t in range(1, num_windows):
                    active_sequence = states[t - 1 : min(t + 2, num_windows)]
                    prob = automata.calculate_path_probability(active_sequence, min_prob=1e-6)
                    scores[t] = 1.0 - prob
                
                y_true = val_labels[config.WINDOW_SIZE:]
                y_scores = scores[1:]
            else:
                _, val_loader_clean = loader_fold.get_fold_dataloaders(
                    val_scaled, val_scaled, val_labels, val_labels
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
                model = load_model(model, checkpoint_path, config.DEVICE)
                model.eval()
                y_probs = predict(model, val_loader_clean, config.DEVICE)
                window_size = len(val_labels) - len(y_probs)
                y_true = val_labels[window_size:]
                y_scores = y_probs
                
            all_scores.extend(y_scores)
            all_labels.extend(y_true)
            
    else:  # BATADAL
        loader = DataLoader("BATADAL")
        path = config.BATADAL_PATH
        if os.path.isdir(path):
            path = os.path.join(path, "batadal_training_2.csv")
        raw_data = loader.load_batadal(path)
        scaled_data, _ = loader.preprocess(raw_data)
        labels = loader.get_labels(raw_data)
        
        train_data, val_data, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
        train_labels, val_labels, test_labels = loader.split_chronological(pd.Series(labels))
        
        if model_type == "Automata":
            automata = TimeSeriesAutomata(alphabet_size=config.ALPHABET_SIZE, word_size=4)
            automata.fit(train_data.values, window_size=config.WINDOW_SIZE)
            
            states = automata.generate_states(test_data.values, window_size=config.WINDOW_SIZE)
            num_windows = len(states)
            scores = np.zeros(num_windows)
            for t in range(1, num_windows):
                active_sequence = states[t - 1 : min(t + 2, num_windows)]
                prob = automata.calculate_path_probability(active_sequence, min_prob=1e-6)
                scores[t] = 1.0 - prob
                
            y_true = test_labels.values[config.WINDOW_SIZE:]
            y_scores = scores[1:]
        else:
            _, val_loader, test_loader_clean = loader.get_dataloaders(
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
            model = load_model(model, checkpoint_path, config.DEVICE)
            model.eval()
            y_probs = predict(model, test_loader_clean, config.DEVICE)
            window_size = len(test_labels) - len(y_probs)
            y_true = test_labels.values[window_size:]
            y_scores = y_probs
            
        all_scores = y_scores
        all_labels = y_true
        
    return np.array(all_scores), np.array(all_labels)

def main():
    print("="*80)
    print("RUNNING ACADEMIC FIGURES VISUALIZATION PIPELINE")
    print("="*80)
    
    from scripts.run_experiments import get_model_predictions
    models = ["Automata", "LSTM", "GRU", "CNN"]
    datasets = ["SKAB", "BATADAL"]
    seed = 42
    
    # 1. Generate Confusion Matrices and ROC/PR Curves for each dataset
    for dataset in datasets:
        print(f"\nProcessing classifications for {dataset} (Seed {seed})...")
        preds_dict = {}
        scores_dict = {}
        labels_preds_dict = {}
        labels_scores_dict = {}
        
        for model in models:
            print(f"  Fetching prediction pairs for {model}...")
            # For confusion matrices
            y_pred, y_true_bin = get_model_predictions(model, seed, dataset)
            preds_dict[model] = y_pred
            labels_preds_dict[model] = y_true_bin
            
            # For ROC/PR curves
            y_score, y_true_score = get_model_scores(model, seed, dataset)
            scores_dict[model] = y_score
            labels_scores_dict[model] = y_true_score
            
        # Plot Confusion Matrices Grid
        cm_path = f"results/plots/{dataset}_confusion_matrices.png"
        plot_confusion_matrices(preds_dict, labels_preds_dict, cm_path)
        
        # Plot ROC/PR curves
        roc_prefix = f"results/plots/{dataset}"
        plot_roc_pr_curves(scores_dict, labels_scores_dict, roc_prefix)
        
    # 2. Plot Sensitivity Heatmaps
    print("\nProcessing parameter sensitivity heatmaps...")
    sens_path = "results/plots/parameter_sensitivity_heatmaps.png"
    plot_sensitivity_heatmaps(sens_path)
    
    # 3. Plot Automata transition matrix heatmap
    print("\nProcessing symbolic Automata transition heatmap...")
    trans_path = "results/plots/automata_transition_matrix.png"
    plot_automata_transitions(trans_path)
    
    print("\n" + "="*80)
    print("VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()


