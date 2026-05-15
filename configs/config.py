import torch

class Config:
    SEEDS = [42, 123, 2026, 7, 999]
    BATCH_SIZE = 32
    MAX_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    LEARNING_RATE = 0.001
    
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    WINDOW_SIZE = 10
    ALPHABET_SIZE = 5
    
    SKAB_PATH = "data/skab"
    BATADAL_PATH = "data/batadal"
    
    PCA_COMPONENTS = 1
    
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
