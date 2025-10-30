import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_one_epoch(model, loader, optimizer, criterion, device, edge_index, edge_attr):
    model.train()
    total_loss = 0
    edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
    
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb, edge_index, edge_attr)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device, edge_index, edge_attr, target_scaler):
    model.eval()
    y_true, y_pred = [], []
    edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
    
    with torch.no_grad():
        for Xb, yb in loader:
            preds = model(Xb.to(device), edge_index, edge_attr)
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_true_orig = target_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    # Return metrics on original scale
    return rmse, mae, r2

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience, self.delta = patience, delta
        self.best_score, self.counter, self.early_stop, self.best_state = None, 0, False, None

    def __call__(self, val_score, model):
        improved = False
        if self.best_score is None or val_score < self.best_score - self.delta:
            self.best_score = val_score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter, improved = 0, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved

    def load_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)