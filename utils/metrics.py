import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def save_results(results, filename):
    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame([results])
    path = os.path.join('results/metrics', filename)
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)
