import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def calculate_metrics(y_true, y_pred):
    """
    Computes standard evaluation metrics: Accuracy, Precision, Recall, and F1-score.
    """
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy)
    }

def save_results(results, filename):
    """
    Saves metrics dictionary to a CSV file in a leakage-free persistent metrics directory.
    """
    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame([results])
    path = os.path.join('results/metrics', filename)
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

