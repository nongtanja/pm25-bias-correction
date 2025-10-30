import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

def create_dataset_with_neighbor_features(
    df, station_list, base_feature_cols, pm25_feature_cols, 
    pm25_neighbor_dict, seq_len=3, target_col='PM2.5_scaled', verbose=True
):
    """
    Creates dataset using base features from target station and
    averaged PM2.5 features from neighbors.
    """
    timestamps = sorted(df['timestamp'].unique())
    num_samples = len(timestamps) - seq_len + 1
    num_stations = len(station_list)
    num_base_features = len(base_feature_cols)
    num_pm25_features = len(pm25_feature_cols)
    num_total_features = num_base_features + num_pm25_features

    if verbose:
        print(f"\n💾 Creating neighbor-aware dataset:")
        print(f"  Samples: {num_samples}, Seq len: {seq_len}")
        print(f"  Total features: {num_total_features} = {num_base_features} (base) + {num_pm25_features} (PM2.5 neighbors)")

    X = np.zeros((num_samples, seq_len, num_stations, num_total_features), dtype=np.float32)
    y = np.zeros((num_samples, num_stations, 1), dtype=np.float32)
    
    # Pre-find bias_aod index
    try:
        bias_aod_index = base_feature_cols.index('bias_aod')
    except ValueError:
        bias_aod_index = -1

    for i in range(num_samples):
        seq_ts = timestamps[i:i + seq_len]
        df_seq = df[df['timestamp'].isin(seq_ts)]
        
        for t, ts in enumerate(seq_ts):
            df_t = df_seq[df_seq['timestamp'] == ts].set_index('Station').reindex(station_list)
            df_t = df_t.fillna(0)
            
            # Global average PM2.5 features (for fallback)
            global_pm25_features = df_t[pm25_feature_cols].mean(axis=0).values
            global_pm25_series = pd.Series(global_pm25_features, index=pm25_feature_cols)
            
            for station_idx, station in enumerate(station_list):
                # 1. Base features from self
                base_values = df_t.loc[station, base_feature_cols].values
                
                # 2. PM2.5 features from neighbors
                neighbors = pm25_neighbor_dict.get(station_idx, [])
                
                if len(neighbors) > 0:
                    neighbor_stations = [station_list[n] for n in neighbors]
                    neighbor_pm25_features = df_t.loc[neighbor_stations, pm25_feature_cols].mean(axis=0)
                    pm25_values = neighbor_pm25_features.values
                else:
                    # No neighbors -> use global average
                    neighbor_pm25_features = global_pm25_series
                    pm25_values = global_pm25_features
                
                # 3. Calculate new bias_aod
                if bias_aod_index != -1:
                    current_aod = df_t.loc[station, 'AOD']
                    neighbor_pm25_lag1 = neighbor_pm25_features['PM25_lag1']
                    new_bias_aod = current_aod - neighbor_pm25_lag1
                    base_values[bias_aod_index] = new_bias_aod
                
                # 4. Combine features
                combined_features = np.concatenate([base_values, pm25_values])
                X[i, t, station_idx] = combined_features
                
                # 5. Target
                if t == seq_len - 1:
                    y[i, station_idx] = df_t.loc[station, target_col]
    
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        if verbose:
            print(f"⚠️ Warning: {nan_count} NaN values found in X (filling with 0)")
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
    
    if verbose:
        print(f"✅ Dataset created: X shape {X.shape}, y shape {y.shape}")
    
    return torch.tensor(X), torch.tensor(y)

class STGDataset(Dataset):
    def __init__(self, X, y): 
        self.X, self.y = X, y
    def __len__(self): 
        return self.X.shape[0]
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]