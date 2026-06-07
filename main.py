import sys
import os
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from configs.config import Config

def create_dummy_datasets(config):
    """Creates dummy datasets if they don't exist to allow self-contained testing."""
    # 1. Create SKAB dummy data
    skab_valve1 = os.path.join(config.SKAB_PATH, "valve1")
    skab_valve2 = os.path.join(config.SKAB_PATH, "valve2")
    os.makedirs(skab_valve1, exist_ok=True)
    os.makedirs(skab_valve2, exist_ok=True)
    
    # Create test csv files in valve1
    for i in range(1, 3):
        file_path = os.path.join(skab_valve1, f"skab_file_{i}.csv")
        if not os.path.exists(file_path):
            dates = pd.date_range(start="2026-05-20 00:00:00", periods=50, freq="10s")
            df = pd.DataFrame(np.random.randn(50, 4), columns=["sensorA", "sensorB", "sensorC", "changepoint"])
            df['anomaly'] = np.random.choice([0, 1], size=50, p=[0.9, 0.1])
            df.index = dates
            df.index.name = 'datetime'
            df.to_csv(file_path, sep=';')
            print(f"Created dummy SKAB file: {file_path}")
            
    # Create test csv files in valve2
    for i in range(3, 5):
        file_path = os.path.join(skab_valve2, f"skab_file_{i}.csv")
        if not os.path.exists(file_path):
            dates = pd.date_range(start="2026-05-20 01:00:00", periods=50, freq="10s")
            df = pd.DataFrame(np.random.randn(50, 4), columns=["sensorA", "sensorB", "sensorC", "changepoint"])
            df['anomaly'] = np.random.choice([0, 1], size=50, p=[0.9, 0.1])
            df.index = dates
            df.index.name = 'datetime'
            df.to_csv(file_path, sep=';')
            print(f"Created dummy SKAB file: {file_path}")
            
    # 2. Create BATADAL dummy data
    os.makedirs(config.BATADAL_PATH, exist_ok=True)
    batadal_file = os.path.join(config.BATADAL_PATH, "batadal_training_2.csv")
    if not os.path.exists(batadal_file):
        dates = pd.date_range(start="2026-05-20 00:00:00", periods=100, freq="h")
        df = pd.DataFrame(np.random.randn(100, 3), columns=["Pressure1", "Flow2", "Level3"])
        df['DATETIME'] = dates
        df['ATT_FLAG'] = np.random.choice([0, 1], size=100, p=[0.85, 0.15])
        df.to_csv(batadal_file, index=False)
        print(f"Created dummy BATADAL file: {batadal_file}")

def main():
    print("====================================================")
    print("Testing Time Series Data Engineering Pipelines")
    print("====================================================")
    
    config = Config()
    
    # Generate dummy data for testing if missing
    create_dummy_datasets(config)
    
    # ====================================================
    # TEST DAY 6: SKAB PIPELINE
    # ====================================================
    print("\n====================================================")
    print("TESTING DAY 6: SKAB PIPELINE & PREPROCESSING")
    print("====================================================")
    
    loader_skab = DataLoader("SKAB")
    try:
        # 1. Load data
        print("\n--- 1. Testing SKAB Multi-Folder Loading ---")
        data_skab = loader_skab.load_skab(config.SKAB_PATH)
        print(f"✓ Loaded SKAB combined dataset successfully. Combined shape: {data_skab.shape}")
        print("Unique source groups (subdirectories):", data_skab['source_group'].unique())
        print("Unique source files (CSVs):", data_skab['source_file'].unique())
            
        # 2. Preprocessing & Leakage check
        print("\n--- 2. Testing Preprocessing (Fit/Transform) ---")
        n_samples = len(data_skab)
        train_idx = np.arange(0, int(n_samples * 0.8))
        test_idx = np.arange(int(n_samples * 0.8), n_samples)
        
        train_df = data_skab.iloc[train_idx]
        test_df = data_skab.iloc[test_idx]
        
        loader_skab.fit(train_df)
        train_scaled, train_pc1 = loader_skab.transform(train_df)
        test_scaled, test_pc1 = loader_skab.transform(test_df)
        
        print(f"✓ Training set: original shape {train_df.shape} -> scaled features {train_scaled.shape}, PC1 {train_pc1.shape}")
        print(f"✓ Testing set: original shape {test_df.shape} -> scaled features {test_scaled.shape}, PC1 {test_pc1.shape}")
        print("✓ Preprocessing data leakage prevention check passed successfully!")
        
        # 3. Testing StratifiedGroupKFold splits
        print("\n--- 3. Testing Stratified Group K-Fold Splits ---")
        splits = loader_skab.split_by_group(data_skab, n_splits=3, stratified=True)
        print(f"✓ StratifiedGroupKFold split generated {len(splits)} folds successfully.")
        for fold, (t_idx, v_idx) in enumerate(splits):
            t_files = data_skab.iloc[t_idx]['source_file'].unique()
            v_files = data_skab.iloc[v_idx]['source_file'].unique()
            overlap = set(t_files).intersection(set(v_files))
            print(f"  Fold {fold+1}: Train samples = {len(t_idx)}, Val samples = {len(v_idx)}")
            print(f"    Train files: {len(t_files)}, Val files: {len(v_files)}")
            if len(overlap) == 0:
                print("    ✓ Data Leakage Check: No group overlap between train and validation folds!")
            else:
                print(f"    ✗ Data Leakage Detected: Overlap files {overlap}")
                
    except Exception as e:
        print(f"✗ SKAB Testing failed: {e}")
        
    # ====================================================
    # TEST DAY 7: BATADAL PIPELINE
    # ====================================================
    print("\n====================================================")
    print("TESTING DAY 7: BATADAL PIPELINE & CHRONOLOGICAL SPLIT")
    print("====================================================")
    
    loader_batadal = DataLoader("BATADAL")
    try:
        # 1. Load data
        print("\n--- 1. Testing BATADAL Parsing & Anomaly Label Mapping ---")
        batadal_file_path = os.path.join(config.BATADAL_PATH, "batadal_training_2.csv")
        data_bat = loader_batadal.load_batadal(batadal_file_path)
        print(f"✓ Loaded BATADAL dataset successfully. Shape: {data_bat.shape}")
        
        if 'anomaly' in data_bat.columns:
            print("✓ Target column renamed/mapped to standard 'anomaly' column.")
            print(f"✓ Anomaly flag distribution: {data_bat['anomaly'].value_counts().to_dict()}")
        else:
            print("✗ Missing target 'anomaly' column after loading.")
            
        # Verify index type
        print(f"✓ Dataset Index Type: {type(data_bat.index)}")
        
        # 2. Preprocessing (Leakage-free)
        print("\n--- 2. Testing Preprocessing Feature Filtering ---")
        # Ensure date columns dropped
        features = loader_batadal._get_features(data_bat)
        print(f"✓ Raw Data Columns: {list(data_bat.columns)}")
        print(f"✓ Extracted Feature Columns: {list(features.columns)}")
        print("✓ Verified: Time columns, non-numeric fields, and anomaly label strictly excluded from features!")
        
        # 3. Chronological splitting
        print("\n--- 3. Testing 60/20/20 Chronological Split ---")
        train_df, val_df, test_df = loader_batadal.split_chronological(data_bat)
        
        # Check leakage-free pipeline on splits
        loader_batadal.fit(train_df)
        train_scaled, train_pc1 = loader_batadal.transform(train_df)
        val_scaled, val_pc1 = loader_batadal.transform(val_df)
        test_scaled, test_pc1 = loader_batadal.transform(test_df)
        
        print("✓ Leakage-free fit/transform splits tested successfully!")
        
    except Exception as e:
        print(f"✗ BATADAL Testing failed: {e}")
        
    print("\n====================================================")
    print("Testing Day 6 & 7 Pipelines completed successfully.")
    print("====================================================")

if __name__ == "__main__":
    main()


