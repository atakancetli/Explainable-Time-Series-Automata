import torch
import numpy as np
import os
import copy
import logging
import random

def set_seed(seed: int):
    """
    Enforces strict deterministic seeding for Python, Numpy, and PyTorch (CPU and GPU/MPS).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping to stop the training when the validation loss does not improve
    after a certain number of epochs. Also supports model checkpointing by saving
    the best weights.
    """
    def __init__(self, patience=5, delta=0, checkpoint_path=None, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            checkpoint_path (str): File path to save the best model checkpoint.
                            Default: None
            verbose (bool): If True, logs a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_state_dict = None
        
        self.logger = logging.getLogger("EarlyStopping")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def __call__(self, val_loss, model=None):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model state dictionary when validation loss decreases.
        """
        if self.verbose and val_loss < self.val_loss_min:
            self.logger.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving best state...")
            
        self.val_loss_min = val_loss
        
        if model is not None:
            # Save a deep copy of the weights in memory
            self.best_state_dict = copy.deepcopy(model.state_dict())
            # If checkpoint path is provided, save to disk
            if self.checkpoint_path is not None:
                if os.path.dirname(self.checkpoint_path):
                    os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)

    def load_best_weights(self, model):
        """
        Loads the best saved weights back into the model.
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                self.logger.info("Loaded best model weights from in-memory state.")
        elif self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path))
            if self.verbose:
                self.logger.info(f"Loaded best model weights from checkpoint: {self.checkpoint_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def predict(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in dataloader:
            if isinstance(x, list): x = x[0]
            x = x.to(device)
            output = model(x)
            preds.extend(output.cpu().numpy())
    return np.array(preds)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
