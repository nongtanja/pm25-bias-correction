import joblib
import torch
import os

def save_scalers(feature_scaler, target_scaler, path="models"):
    """Saves scalers to disk."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(path, "feature_scaler.joblib"))
    joblib.dump(target_scaler, os.path.join(path, "target_scaler.joblib"))
    print(f"Scalers saved to {path}/")

def save_stgnn_model(model, path="models", filename="stgnn_best.pth"):
    """Saves STGNN model state dict."""
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, filename)
    torch.save(model.state_dict(), model_path)
    print(f"STGNN model saved to {model_path}")

def save_rf_model(model, path="models", filename="rf_baseline.joblib"):
    """Saves Random Forest model."""
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, filename)
    joblib.dump(model, model_path)
    print(f"RF model saved to {model_path}")