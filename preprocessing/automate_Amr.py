# automate_Amr.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

SENSOR_COLS_TO_DROP = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
RUL_CLIP = 125
ROLLING_WINDOWS = [5, 10]
COLUMN_NAMES = [
    'unit_nr', 'time_cycles', 'op_1', 'op_2', 'op_3',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
]

def load_data(train_path, test_path, rul_path):
    """Load train, test, and ground truth RUL datasets."""
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=['RUL'])
    return train_df, test_df, rul_df

def drop_constant_sensors(df, cols_to_drop):
    """Drop non-informative or constant sensors."""
    return df.drop(columns=cols_to_drop)

def derive_rul(train_df, clip_value):
    """Calculate RUL target for train set and clip it."""
    max_cycles = train_df.groupby('unit_nr')['time_cycles'].max()
    train_df = train_df.merge(max_cycles.rename('max_cycle'), on='unit_nr')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycles']
    train_df.drop(columns=['max_cycle'], inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=clip_value)
    return train_df

def add_rolling_features(df, sensor_cols, windows):
    """Compute rolling mean and rolling std for sensors grouped by unit_nr."""
    features_df = df.copy()
    for window in windows:
        # Rolling mean
        rolling_mean = df.groupby('unit_nr')[sensor_cols].rolling(window=window).mean().reset_index(level=0, drop=True)
        rolling_mean.columns = [f"{col}_roll_mean_{window}" for col in sensor_cols]
        
        # Rolling std
        rolling_std = df.groupby('unit_nr')[sensor_cols].rolling(window=window).std().reset_index(level=0, drop=True)
        rolling_std.columns = [f"{col}_roll_std_{window}" for col in sensor_cols]
        
        features_df = pd.concat([features_df, rolling_mean, rolling_std], axis=1)
    
    # Drop any NaNs resulting from rolling window calculations
    features_df.dropna(inplace=True)
    features_df.reset_index(drop=True, inplace=True)
    return features_df

def normalize(train_df, test_df, feature_cols):
    """Fit MinMaxScaler on train features and transform both datasets."""
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_scaled, test_scaled

def prepare_test_labels(test_df, rul_df):
    """Associate ground truth RUL with the last cycle of each unit in the test set."""
    test_last_cycle = test_df.groupby('unit_nr').last().reset_index()
    rul_df_with_id = rul_df.copy()
    rul_df_with_id['unit_nr'] = rul_df_with_id.index + 1
    test_final = test_last_cycle.merge(rul_df_with_id, on='unit_nr')
    return test_final

def save_outputs(train_df, test_df, output_dir):
    """Export preprocessed train and test dataframes to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    train_output_path = os.path.join(output_dir, "FD001_preprocessing_train.csv")
    test_output_path = os.path.join(output_dir, "FD001_preprocessing_test.csv")
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    print(f"Preprocessed train dataset saved to {train_output_path}")
    print(f"Preprocessed test dataset saved to {test_output_path}")

def run_pipeline(train_path, test_path, rul_path, output_dir):
    """Execute the full preprocessing pipeline."""
    print("Starting preprocessing pipeline...")
    
    # Load raw data
    train_df, test_df, rul_df = load_data(train_path, test_path, rul_path)
    print(f"Raw train shape: {train_df.shape}, test shape: {test_df.shape}")
    
    # Drop constant sensors
    train_df = drop_constant_sensors(train_df, SENSOR_COLS_TO_DROP)
    test_df = drop_constant_sensors(test_df, SENSOR_COLS_TO_DROP)
    
    # Derive RUL for train
    train_df = derive_rul(train_df, RUL_CLIP)
    
    # Features lists
    remaining_sensors = [col for col in train_df.columns if col.startswith('s')]
    
    # Add rolling features
    train_df = add_rolling_features(train_df, remaining_sensors, ROLLING_WINDOWS)
    test_df = add_rolling_features(test_df, remaining_sensors, ROLLING_WINDOWS)
    
    # Normalize features
    exclude_cols = ['unit_nr', 'time_cycles', 'RUL']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    train_df, test_df = normalize(train_df, test_df, feature_cols)
    
    # Prepare test labels (RUL)
    test_final = prepare_test_labels(test_df, rul_df)
    
    # Save datasets
    save_outputs(train_df, test_final, output_dir)
    print("Preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline(
        train_path="FD001_raw/train_FD001.txt",
        test_path="FD001_raw/test_FD001.txt",
        rul_path="FD001_raw/RUL_FD001.txt",
        output_dir="preprocessing/FD001_preprocessing"
    )
