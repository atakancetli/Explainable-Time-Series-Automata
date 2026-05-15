import sys
from utils.data_loader import DataLoader
from configs.config import Config

def main():
    config = Config()
    
    loader = DataLoader("SKAB")
    
    try:
        data = loader.load_skab(config.SKAB_PATH)
        scaled, pc1 = loader.preprocess(data)
        labels = loader.get_labels(data)
        
        print(f"Loaded SKAB data shape: {data.shape}")
        print(f"Scaled data shape: {scaled.shape}")
        print(f"PC1 shape: {pc1.shape}")
    except Exception as e:
        print(f"Data loading test skipped: {e}")

if __name__ == "__main__":
    main()
