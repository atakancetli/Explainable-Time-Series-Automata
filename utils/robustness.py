import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.metrics import f1_score
from utils.metrics import calculate_metrics

def inject_gaussian_noise(data, scale=0.1):
    """
    Injects zero-mean Gaussian noise into the numerical feature matrix.
    
    Parameters:
    -----------
    data : np.ndarray or pd.DataFrame
        The numerical features to perturb.
    scale : float
        The standard deviation of the Gaussian noise.
        
    Returns:
    --------
    perturbed_data : np.ndarray or pd.DataFrame
        The data with added Gaussian noise.
    """
    if scale <= 0:
        return data
        
    if isinstance(data, pd.DataFrame):
        perturbed = data.copy()
        numeric_cols = perturbed.select_dtypes(include=[np.number]).columns
        noise = np.random.normal(loc=0.0, scale=scale, size=perturbed[numeric_cols].shape)
        orig_std = perturbed[numeric_cols].values.std()
        perturbed[numeric_cols] += noise
        new_std = perturbed[numeric_cols].values.std()
        print(f"[NoiseVerify] DataFrame | scale={scale} | orig_std={orig_std:.6f} | new_std={new_std:.6f} | delta={abs(new_std - orig_std):.6f}")
        return perturbed
    elif isinstance(data, np.ndarray):
        noise = np.random.normal(loc=0.0, scale=scale, size=data.shape)
        orig_std = data.std()
        perturbed = data + noise
        new_std = perturbed.std()
        print(f"[NoiseVerify] ndarray | scale={scale} | orig_std={orig_std:.6f} | new_std={new_std:.6f} | delta={abs(new_std - orig_std):.6f}")
        return perturbed
    else:
        # If list/tuple, convert to np.ndarray
        arr = np.array(data, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=scale, size=arr.shape)
        orig_std = arr.std()
        perturbed = arr + noise
        new_std = perturbed.std()
        print(f"[NoiseVerify] other | scale={scale} | orig_std={orig_std:.6f} | new_std={new_std:.6f} | delta={abs(new_std - orig_std):.6f}")
        return perturbed

def calculate_unseen_metrics(automata, train_data, test_data, train_labels, test_labels, window_size=10, threshold=0.01):
    """
    Calculates detection rates and Levenshtein mapping accuracy metrics for unseen test states.
    
    Parameters:
    -----------
    automata : TimeSeriesAutomata
        The trained Automata model.
    train_data : np.ndarray or pd.DataFrame
        Training features.
    test_data : np.ndarray or pd.DataFrame
        Testing features.
    train_labels : np.ndarray
        Ground truth binary training labels.
    test_labels : np.ndarray
        Ground truth binary testing labels.
    window_size : int
        The sliding window size.
    threshold : float
        The path probability threshold for anomaly detection.
        
    Returns:
    --------
    metrics : dict
        A dictionary containing 'detection_rate' and 'mapping_accuracy'.
    """
    # 1. Convert inputs to numpy arrays
    t_data = train_data.values if isinstance(train_data, pd.DataFrame) else np.asarray(train_data)
    te_data = test_data.values if isinstance(test_data, pd.DataFrame) else np.asarray(test_data)
    t_labels = train_labels.values if isinstance(train_labels, pd.Series) else np.asarray(train_labels)
    te_labels = test_labels.values if isinstance(test_labels, pd.Series) else np.asarray(test_labels)
    
    # 2. Extract training states and their majority label profiles
    train_states = automata.generate_states(t_data, window_size=window_size)
    train_labels_aligned = t_labels[window_size - 1:]
    
    state_to_labels = defaultdict(list)
    for s, lbl in zip(train_states, train_labels_aligned):
        state_to_labels[s].append(lbl)
        
    state_majority_label = {}
    for s, lbls in state_to_labels.items():
        state_majority_label[s] = int(np.round(np.mean(lbls)))
        
    # 3. Extract test states and match with test labels
    test_states = automata.generate_states(te_data, window_size=window_size)
    test_labels_aligned = te_labels[window_size - 1:]
    
    # Predict anomalies for test set using automata
    # Note: automata predictions have length len(test_states), with predictions[0] = 0 (normal)
    predictions = automata.predict_anomaly(te_data, window_size=window_size, threshold=threshold)
    
    unseen_count = 0
    correct_anomaly_flag = 0
    correct_mappings = 0
    
    # Training unique vocabulary
    seen_vocabulary = set(automata.unique_states)
    
    for idx, s_test in enumerate(test_states):
        # We check if the current test state was unseen during training
        if s_test not in seen_vocabulary:
            unseen_count += 1
            
            # Metric A: Detection Rate (did we flag it as anomaly?)
            # Map test state index to predict_anomaly index (which is directly idx)
            is_anomaly_predicted = int(predictions[idx])
            if is_anomaly_predicted == 1:
                correct_anomaly_flag += 1
                
            # Metric B: Mapping Accuracy
            s_mapped = automata._map_unseen_state(s_test)
            mapped_lbl = state_majority_label.get(s_mapped, 0)
            true_lbl = int(test_labels_aligned[idx])
            if mapped_lbl == true_lbl:
                correct_mappings += 1
                
    if unseen_count > 0:
        det_rate = correct_anomaly_flag / unseen_count
        map_acc = correct_mappings / unseen_count
    else:
        det_rate = 1.0 # default to 1.0 if no unseen states exist
        map_acc = 1.0
        
    return {
        "detection_rate": det_rate,
        "mapping_accuracy": map_acc,
        "unseen_count": unseen_count
    }
