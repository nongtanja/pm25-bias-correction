import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import from our new src modules
from src.data_processing import prepare_data
from src.graph_utils import create_graph_structure, create_pm25_neighbor_dict
from src.dataset import create_dataset_with_neighbor_features, STGDataset
from src.models.stgnn import STGNNModel
from src.models.rf_baseline import train_evaluate_rf_baseline
from src.training import train_one_epoch, evaluate, EarlyStopping
from src.utils import save_scalers, save_stgnn_model, save_rf_model

def main(args):
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)

    # --- 2. Load and Prepare Data ---
    print(f"Loading data from {args.data_path}")
    df_raw = pd.read_csv(args.data_path)
    df_raw = df_raw.drop(columns=[c for c in ['year','month','day','day_of_year','week_of_year','weekday'] 
                                 if c in df_raw.columns], errors='ignore')
    
    df, base_feature_cols, pm25_feature_cols = prepare_data(df_raw, verbose=True)
    coords_df = df.drop_duplicates('Station')[['Station','latitude','longitude']]
    
    # Get station list
    all_dates = df['timestamp'].unique()
    pivot = df.pivot_table(index='timestamp', columns='Station', values='PM2.5', aggfunc='count')
    station_list = sorted(pivot.columns[pivot.notna().sum() == len(all_dates)].tolist())
    df = df[df['Station'].isin(station_list)].copy()
    print(f"Filtered to {len(station_list)} stations with complete data.")

    # --- 3. Data Splitting and Scaling ---
    test_start_date = pd.to_datetime(args.test_date)
    val_start_date = pd.to_datetime(args.val_date)
    
    train_mask = df['timestamp'] < val_start_date
    val_mask = (df['timestamp'] >= val_start_date) & (df['timestamp'] < test_start_date)
    test_mask = df['timestamp'] >= test_start_date
   
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    
    feature_scaler = MinMaxScaler()
    target_scaler = StandardScaler()
    
    feature_cols = base_feature_cols + pm25_feature_cols
    
    df_train[feature_cols] = feature_scaler.fit_transform(df_train[feature_cols])
    df_train['PM2.5_scaled'] = target_scaler.fit_transform(df_train[['PM2.5']])
    
    df_val[feature_cols] = feature_scaler.transform(df_val[feature_cols])
    df_val['PM2.5_scaled'] = target_scaler.transform(df_val[['PM2.5']])
    
    df_test[feature_cols] = feature_scaler.transform(df_test[feature_cols])
    df_test['PM2.5_scaled'] = target_scaler.transform(df_test[['PM2.5']])
    
    df_scaled = pd.concat([df_train, df_val, df_test]).sort_values(['Station', 'timestamp']).reset_index(drop=True)

    # --- 4. Create Graph and Dataset ---
    edge_index, edge_attr = create_graph_structure(
        coords_df, df_scaled, feature_cols, station_list, 
        test_start_date=val_start_date,
        k=args.k_gnn, 
        alpha=args.alpha_gnn,
        verbose=True
    )
    
    pm25_neighbor_dict = create_pm25_neighbor_dict(
        coords_df, station_list, 
        k_pm25=args.k_pm25, 
        verbose=True
    )
    
    X, y = create_dataset_with_neighbor_features(
        df_scaled, station_list, 
        base_feature_cols, 
        pm25_feature_cols, 
        pm25_neighbor_dict, 
        seq_len=args.seq_len, 
        target_col='PM2.5_scaled', 
        verbose=True
    )

    # --- 5. Split Sequences ---
    all_timestamps = sorted(df_scaled['timestamp'].unique())
    val_split_idx = np.searchsorted(pd.to_datetime(all_timestamps[args.seq_len - 1:]), val_start_date)
    test_split_idx = np.searchsorted(pd.to_datetime(all_timestamps[args.seq_len - 1:]), test_start_date)

    X_train, y_train = X[:val_split_idx], y[:val_split_idx]
    X_val, y_val = X[val_split_idx:test_split_idx], y[val_split_idx:test_split_idx]
    X_test, y_test = X[test_split_idx:], y[test_split_idx:]

    print(f"Train sequences: {X_train.shape[0]}")
    print(f"Val sequences: {X_val.shape[0]}")
    print(f"Test sequences: {X_test.shape[0]}")

    # --- 6. Model Training ---
    
    if args.model == 'stgnn':
        print("\n--- 🧠 Training STGNN Model ---")
        
        train_loader = DataLoader(STGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(STGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(STGDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
        
        num_total_features = len(base_feature_cols) + len(pm25_feature_cols)
        model = STGNNModel(
            num_total_features, 
            len(station_list), 
            hidden_dim=args.hidden_dim, 
            dropout=args.dropout
        ).to(device)
        
        print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10, delta=1e-5)
        
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, edge_index, edge_attr)
            val_rmse, val_mae, val_r2 = evaluate(model, val_loader, device, edge_index, edge_attr, target_scaler)
            
            scheduler.step(val_rmse)
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            early_stopping(val_rmse, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("Loading best model weights...")
        early_stopping.load_best_weights(model)
        
        print("\n📊 Final Evaluation (STGNN)")
        train_rmse, train_mae, train_r2 = evaluate(model, train_loader, device, edge_index, edge_attr, target_scaler)
        val_rmse, val_mae, val_r2 = evaluate(model, val_loader, device, edge_index, edge_attr, target_scaler)
        test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device, edge_index, edge_attr, target_scaler)
        
        print(f"Train → RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
        print(f"Val → RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
        print(f"Test → RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")
        
        # Save model
        save_stgnn_model(model, args.model_dir, "stgnn_best.pth")

    elif args.model == 'rf':
        rf_model, rf_metrics = train_evaluate_rf_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test,
            target_scaler,
            base_feature_cols,
            pm25_feature_cols,
            n_estimators=args.rf_trees,
            max_depth=args.rf_depth
        )
        # Save model
        save_rf_model(rf_model, args.model_dir, "rf_baseline.joblib")

    # --- 7. Save Scalers ---
    save_scalers(feature_scaler, target_scaler, args.model_dir)
    print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STGNN or RF model for PM2.5 Bias Correction")
    
    # General args
    parser.add_argument('--model', type=str, required=True, choices=['stgnn', 'rf'], help="Model to train ('stgnn' or 'rf')")
    parser.add_argument('--data_path', type=str, default="data/raw/aod_pm25_thailand_mean_imputed.csv", help="Path to raw data CSV")
    parser.add_argument('--model_dir', type=str, default="models", help="Directory to save trained models and scalers")
    parser.add_argument('--seq_len', type=int, default=3, help="Sequence length for input")
    parser.add_argument('--val_date', type=str, default='2023-01-01', help="Start date for validation set")
    parser.add_argument('--test_date', type=str, default='2024-01-01', help="Start date for test set")

    # Graph structure args
    parser.add_argument('--k_gnn', type=int, default=7, help="k-neighbors for GNN graph structure")
    parser.add_argument('--alpha_gnn', type=float, default=0.3, help="Alpha for combining dist/feature similarity in GNN graph")
    parser.add_argument('--k_pm25', type=int, default=1, help="k-neighbors for PM2.5 neighbor feature engineering")

    # STGNN args
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for STGNN")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for STGNN")
    parser.add_argument('--lr', type=float, default=0.002, help="Learning rate for STGNN")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension for STGNN")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate for STGNN")

    # RF args
    parser.add_argument('--rf_trees', type=int, default=100, help="Number of trees for Random Forest")
    parser.add_argument('--rf_depth', type=int, default=10, help="Max depth for Random Forest")

    args = parser.parse_args()
    main(args)