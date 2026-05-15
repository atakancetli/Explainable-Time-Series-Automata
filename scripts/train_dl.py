import torch
import torch.nn as nn
from configs.config import Config
from models.model_factory import ModelFactory
from utils.data_loader import DataLoader
from utils.train_utils import train_one_epoch, validate, EarlyStopping, save_model
from utils.logger import setup_logger
from utils.visualization import plot_loss
import pandas as pd

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
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    train_losses, val_losses = [], []
    
    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
            
    save_model(model, f"checkpoints/{dataset_name}_{model_type}.pth")
    plot_loss(train_losses, val_losses, f"results/plots/{dataset_name}_{model_type}_loss.png")

if __name__ == "__main__":
    for m in ["LSTM", "CNN"]:
        run_training("SKAB", m)
