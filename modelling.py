# modelling.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Set tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("cmapss-rul-prediction")

# Load preprocessed training data
train_data_path = "preprocessing/FD001_preprocessing/FD001_preprocessing_train.csv"
print(f"Loading preprocessed training data from {train_data_path}...")
train = pd.read_csv(train_data_path)

# Split by unit_nr (NOT random split) to maintain time-series structure and prevent data leakage
train_units = train['unit_nr'].unique()
split_idx = int(len(train_units) * 0.8)
train_set = train[train['unit_nr'].isin(train_units[:split_idx])]
val_set = train[train['unit_nr'].isin(train_units[split_idx:])]

X_train = train_set.drop(['unit_nr', 'time_cycles', 'RUL'], axis=1)
y_train = train_set['RUL']
X_val = val_set.drop(['unit_nr', 'time_cycles', 'RUL'], axis=1)
y_val = val_set['RUL']

print(f"Train features shape: {X_train.shape}, Validation features shape: {X_val.shape}")

# Start MLflow run
with mlflow.start_run():
    # Enable automatic logging for scikit-learn models
    mlflow.sklearn.autolog()
    
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate additional validation metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"Validation Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # Explicitly log additional validation metrics
    mlflow.log_metric("val_mae", mae)
    mlflow.log_metric("val_rmse", rmse)
    mlflow.log_metric("val_r2", r2)

print("Training completed and logged to MLflow successfully!")
