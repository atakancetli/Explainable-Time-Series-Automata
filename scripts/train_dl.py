import torch
import torch.nn as nn
from configs.config import Config
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader
from utils.train_utils import train_one_epoch, validate, EarlyStopping, save_model, predict
from utils.logger import setup_logger
from utils.visualization import plot_loss
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(model, val_loader, val_labels, device):
    """
    Finds the optimal anomaly threshold on the validation set that maximizes the F1-score.
    
    Args:
        model: Trained PyTorch model.
        val_loader: DataLoader for the validation dataset.
        val_labels: Ground truth binary labels for the validation dataset.
        device: The PyTorch device to run model inference on.
        
    Returns:
        float: The threshold (between 0 and 1) that maximizes F1-score.
        float: The maximum F1-score achieved.
    """
    model.eval()
    y_probs = predict(model, val_loader, device)
    
    # Extract ground truth matching the predictions length (sliding window shift)
    if hasattr(val_loader.dataset, 'labels') and val_loader.dataset.labels is not None:
        y_true = val_loader.dataset.labels[val_loader.dataset.window_size:].cpu().numpy()
    else:
        window_size = len(val_labels) - len(y_probs)
        y_true = val_labels[window_size:]
        
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Search threshold range
    for thresh in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    return float(best_thresh), float(best_f1)

def run_training(dataset_name, model_type):
    config = Config()
    logger = setup_logger(f"Train_{model_type}", f"{dataset_name}_{model_type}.log")
    
    loader = DataLoader(dataset_name)
    raw_data = loader.load_skab(config.SKAB_PATH) if dataset_name == "SKAB" else loader.load_batadal(config.BATADAL_PATH)
    scaled_data, _ = loader.preprocess(raw_data)
    labels = loader.get_labels(raw_data)
    
    train_data, val_data, test_data = loader.split_chronological(pd.DataFrame(scaled_data))
    train_labels, val_labels, test_labels = loader.split_chronological(pd.Series(labels))
    
    train_loader, val_loader, test_loader = loader.get_dataloaders(
        train_data.values, val_data.values, test_data.values,
        train_labels.values, val_labels.values, test_labels.values
    )
    
    model = ModelFactory.get_model(model_type, scaled_data.shape[1], config.HIDDEN_SIZE, config.NUM_LAYERS).to(config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    checkpoint_path = f"checkpoints/{dataset_name}_{model_type}_best.pth"
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        checkpoint_path=checkpoint_path,
        verbose=True
    )
    
    train_losses, val_losses = [], []
    
    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
            
    logger.info("Loading best model weights from checkpoint...")
    early_stopping.load_best_weights(model)
    
    save_model(model, f"checkpoints/{dataset_name}_{model_type}.pth")
    
    # Calculate and log best prediction threshold
    best_thresh, best_f1 = find_best_threshold(model, val_loader, val_labels.values, config.DEVICE)
    logger.info(f"Optimal threshold found: {best_thresh:.4f} with Validation F1-score: {best_f1:.4f}")
    
    plot_loss(train_losses, val_losses, f"results/plots/{dataset_name}_{model_type}_loss.png")

if __name__ == "__main__":
    for m in ["LSTM", "CNN"]:
        run_training("SKAB", m)
