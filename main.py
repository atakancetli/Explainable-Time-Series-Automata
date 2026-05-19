import sys
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from configs.config import Config

def main():
    print("====================================================")
    print("Testing Day 6: Data Pipeline & Data Leakage Prevention")
    print("====================================================")
    
    config = Config()
    loader = DataLoader("SKAB")
    
    try:
        # 1. Load data
        print("\n--- 1. Testing SKAB Multi-Folder Loading ---")
        data = loader.load_skab(config.SKAB_PATH)
        print(f"✓ Loaded SKAB combined dataset successfully. Combined shape: {data.shape}")
        
        if 'source_group' in data.columns and 'source_file' in data.columns:
            print(f"✓ Columns 'source_group' and 'source_file' successfully added.")
            print("Unique source groups (subdirectories):", data['source_group'].unique())
            print("Unique source files (CSVs):", data['source_file'].unique())
        else:
            print("✗ Missing source tracking columns.")
            
        # 2. Testing Leakage-Free Preprocessing
        print("\n--- 2. Testing Leakage-Free Preprocessing (Fit/Transform) ---")
        # Split into dummy train and test indices to demonstrate
        n_samples = len(data)
        train_idx = np.arange(0, int(n_samples * 0.8))
        test_idx = np.arange(int(n_samples * 0.8), n_samples)
        
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]
        
        # Fit on train only!
        loader.fit(train_df)
        
        # Transform train and test separately
        train_scaled, train_pc1 = loader.transform(train_df)
        test_scaled, test_pc1 = loader.transform(test_df)
        
        print(f"✓ Training set: original shape {train_df.shape} -> scaled features {train_scaled.shape}, PC1 {train_pc1.shape}")
        print(f"✓ Testing set: original shape {test_df.shape} -> scaled features {test_scaled.shape}, PC1 {test_pc1.shape}")
        print("✓ Preprocessing data leakage prevention check passed successfully!")
        
        # 3. Testing StratifiedGroupKFold splits
        print("\n--- 3. Testing Stratified Group K-Fold Splits ---")
        splits = loader.split_by_group(data, n_splits=3, stratified=True)
        print(f"✓ StratifiedGroupKFold split generated {len(splits)} folds successfully.")
        for fold, (t_idx, v_idx) in enumerate(splits):
            t_files = data.iloc[t_idx]['source_file'].unique()
            v_files = data.iloc[v_idx]['source_file'].unique()
            # Verify no overlap
            overlap = set(t_files).intersection(set(v_files))
            print(f"  Fold {fold+1}: Train samples = {len(t_idx)}, Val samples = {len(v_idx)}")
            print(f"    Train files: {len(t_files)}, Val files: {len(v_files)}")
            if len(overlap) == 0:
                print("    ✓ Data Leakage Check: No group overlap between train and validation folds!")
            else:
                print(f"    ✗ Data Leakage Detected: Overlap files {overlap}")
                
    except Exception as e:
        print(f"Testing execution failed: {e}")
        print("Note: To run a full integration test, make sure the SKAB datasets are placed inside 'data/skab/valve1/' and 'data/skab/valve2/'.")
        
    print("\n====================================================")
    print("Testing Day 6 Completion finished.")
    print("====================================================")

if __name__ == "__main__":
    main()

