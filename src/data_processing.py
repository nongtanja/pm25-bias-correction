import pandas as pd
import numpy as np

def prepare_data(df, verbose=True):
    """
    Cleans and performs feature engineering on the raw dataframe.
    """
    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=['Date', 'station_id'], keep='first').copy()
    if verbose:
        print(f"🔍 Removed {n_before - len(df)} duplicate rows")
    
    df['timestamp'] = pd.to_datetime(df['Date'])
    df['Station'] = df['station_id']

    # Sort by station and time
    df = df.sort_values(['Station', 'timestamp']).reset_index(drop=True)
    
    # ✨ แยก feature เป็น 2 กลุ่ม
    base_feature_cols = ['temp', 'humidity', 'precip', 'pressure', 'windspeed', 'AOD']
    pm25_feature_cols = [] # จะเก็บ features ที่เกี่ยวกับ PM2.5
    
    # Cyclical time encoding (เก็บใน base)
    df['doy'] = df['timestamp'].dt.dayofyear
    df['doy_sin'] = np.sin(2*np.pi*df['doy']/365)
    df['doy_cos'] = np.cos(2*np.pi*df['doy']/365)
    df['month'] = df['timestamp'].dt.month
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    base_feature_cols += ['doy_sin', 'doy_cos', 'month_sin', 'month_cos']

    # PM2.5 lag features (จะย้ายไป pm25_feature_cols)
    for lag in [1, 3, 7]:
        df[f'PM25_lag{lag}'] = df.groupby('Station')['PM2.5'].shift(lag)
        pm25_feature_cols.append(f'PM25_lag{lag}')
    
    # PM2.5 rolling features (จะย้ายไป pm25_feature_cols)
    df['PM25_roll3'] = df.groupby('Station')['PM2.5'].shift(1).transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['PM25_roll7'] = df.groupby('Station')['PM2.5'].shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    pm25_feature_cols += ['PM25_roll3', 'PM25_roll7']
    
    # Bias AOD (เพิ่มชื่อเข้าไปใน list)
    base_feature_cols.append('bias_aod')

    # AOD lag features (เก็บไว้ใน base)
    for lag in [1, 3, 7]:
        df[f'AOD_lag{lag}'] = df.groupby('Station')['AOD'].shift(lag)
        base_feature_cols.append(f'AOD_lag{lag}')

    # --- 🔥 START FIX ---
    # สร้างคอลัมน์ 'bias_aod' ให้เป็นค่าว่าง (NaN) ก่อน
    if 'bias_aod' not in df.columns:
        df['bias_aod'] = np.nan 
    # --- 🔥 END FIX ---

    # Fill missing values
    all_features = base_feature_cols + pm25_feature_cols
    for col in all_features:
        if col in df.columns:
            df[col] = df.groupby('Station')[col].ffill().bfill()
    
    # Final safety fill
    df[all_features] = df[all_features].fillna(0)
    df['PM2.5'] = df['PM2.5'].bfill().ffill()
    
    if verbose:
        print(f"\n📊 Feature Groups:")
        print(f"  Base features (self): {len(base_feature_cols)}")
        print(f"  PM2.5 features (neighbor): {len(pm25_feature_cols)}")
        
    return df, base_feature_cols, pm25_feature_cols