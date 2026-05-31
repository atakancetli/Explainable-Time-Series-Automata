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
