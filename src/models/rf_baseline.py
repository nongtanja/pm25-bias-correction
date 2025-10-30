import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def reshape_for_rf(X, y):
    """
    Reshapes 4D data (S, T, N, F) into RF 2D data (S*N, F).
    Uses only the last time step (t-0).
    """
    S, T, N, F = X.shape
    
    # 1. Select last time step
    X_last_step = X[:, -1, :, :] 
    
    # 2. Flatten S and N: (S, N, F) -> (S*N, F)
    X_flat = X_last_step.reshape(S * N, F).cpu().numpy()
    
    # 3. Flatten target: (S, N, 1) -> (S*N,)
    y_flat = y.reshape(S * N).cpu().numpy()
    
    X_final = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    y_flat = np.nan_to_num(y_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_final, y_flat

def train_evaluate_rf_baseline(X_train, y_train, X_val, y_val, X_test, y_test, 
                               target_scaler, 
                               base_feature_cols, 
                               pm25_feature_cols,
                               n_estimators=100,
                               max_depth=10):
    """
    Trains and evaluates the Random Forest baseline model.
    Returns the trained model and a dictionary of metrics.
    """
    print(f"\n--- 🌳 Training Random Forest Baseline (t-0 only) ---")
    print(f"Params: n_estimators={n_estimators}, max_depth={max_depth}")
    
    X_train_rf, y_train_rf = reshape_for_rf(X_train, y_train)
    X_val_rf, y_val_rf = reshape_for_rf(X_val, y_val)
    X_test_rf, y_test_rf = reshape_for_rf(X_test, y_test)
    
    rf_feature_names = base_feature_cols + pm25_feature_cols
    print(f"\nTotal features in RF model: {len(rf_feature_names)}")

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Fitting RF model on {X_train_rf.shape[0]} samples...")
    rf_model.fit(X_train_rf, y_train_rf)
    
    print("\n📊 RF Feature Importance (Top 15)")
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': rf_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    print(feature_importance_df.head(15).to_string(index=False))

    def eval_rf(model, X_rf, y_rf, scaler):
        preds_scaled = model.predict(X_rf)
        preds_orig = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        true_orig = scaler.inverse_transform(y_rf.reshape(-1, 1)).ravel()
        
        rmse = np.sqrt(mean_squared_error(true_orig, preds_orig))
        mae = mean_absolute_error(true_orig, preds_orig)
        r2 = r2_score(true_orig, preds_orig)
        return rmse, mae, r2

    print("\nEvaluating RF model...")
    train_rmse, train_mae, train_r2 = eval_rf(rf_model, X_train_rf, y_train_rf, target_scaler)
    val_rmse, val_mae, val_r2 = eval_rf(rf_model, X_val_rf, y_val_rf, target_scaler)
    test_rmse, test_mae, test_r2 = eval_rf(rf_model, X_test_rf, y_test_rf, target_scaler)
    
    print("\n📊 Random Forest Baseline Evaluation")
    print(f"RF Train → RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
    print(f"RF Val   → RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
    print(f"RF Test  → RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")
    print("----------------------------")
    
    metrics = {
        "train": {"rmse": train_rmse, "mae": train_mae, "r2": train_r2},
        "val": {"rmse": val_rmse, "mae": val_mae, "r2": val_r2},
        "test": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2}
    }
    
    return rf_model, metrics