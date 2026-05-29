import numpy as np
import scipy.stats as stats

def calculate_mcnemar_test(preds_A, preds_B, y_true):
    """
    Computes McNemar's statistical significance test between two models A and B.
    
    Parameters:
    preds_A (np.ndarray): Binary predictions of model A (0 or 1).
    preds_B (np.ndarray): Binary predictions of model B (0 or 1).
    y_true (np.ndarray): Ground truth binary labels (0 or 1).
    
    Returns:
    dict: contingency_table [[a, b], [c, d]], statistic, p_value, and significance flag.
    """
    # Cast to numpy arrays
    preds_A = np.array(preds_A).astype(int)
    preds_B = np.array(preds_B).astype(int)
    y_true = np.array(y_true).astype(int)
    
    if len(preds_A) != len(preds_B) or len(preds_A) != len(y_true):
        raise ValueError("All inputs must have identical length.")
        
    # Correctness arrays (True/False)
    correct_A = (preds_A == y_true)
    correct_B = (preds_B == y_true)
    
    # 2x2 Contingency Table
    # A \ B    Correct   Incorrect
    # Correct    a          b
    # Incorrect  c          d
    a = np.sum(correct_A & correct_B)
    b = np.sum(correct_A & ~correct_B)  # A right, B wrong
    c = np.sum(~correct_A & correct_B)  # A wrong, B right
    d = np.sum(~correct_A & ~correct_B)
    
    contingency_table = [
        [int(a), int(b)],
        [int(c), int(d)]
    ]
    
    # McNemar's test
    # b + c is the number of discordant pairs
    n_discordant = b + c
    
    if n_discordant == 0:
        statistic = 0.0
        p_value = 1.0
    elif n_discordant < 25:
        # Use exact binomial test
        statistic = float(min(b, c))
        # Cumulative probability from binomial distribution under H0: p=0.5
        p_value = float(2.0 * stats.binom.cdf(min(b, c), n_discordant, 0.5))
    else:
        # Use Chi-Squared test with Yates's continuity correction
        statistic = float(((abs(b - c) - 1) ** 2) / n_discordant)
        p_value = float(1.0 - stats.chi2.cdf(statistic, 1))
        
    significant = p_value < 0.05
    
    return {
        "contingency_table": contingency_table,
        "statistic": statistic,
        "p_value": p_value,
        "significant": bool(significant)
    }
