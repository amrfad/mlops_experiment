# modelling_tuning.py
import os
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# Verify that credentials exist
username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

if not username or not token:
    raise ValueError("DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set in .env file!")

# Set environment variables for DagsHub authentication before importing/init
os.environ["DAGSHUB_USERNAME"] = username
os.environ["DAGSHUB_TOKEN"] = token

import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Initialize DagsHub tracking integration
print(f"Initializing DagsHub MLflow tracking for repository: {username}/cmapss-mlops...")
dagshub.init(repo_owner=username, repo_name='cmapss-mlops', mlflow=True)

# Set experiment name
mlflow.set_experiment("cmapss-rul-tuning")

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

# Define hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting hyperparameter tuning using RandomizedSearchCV (with 3-fold CV and 10 iterations)...")
# Note: we use verbose=2 on RandomizedSearchCV to provide full visibility of the tuning progress
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='neg_mean_absolute_error',
    random_state=42,
    verbose=2,
    n_jobs=-1
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"Hyperparameter tuning completed. Best parameters found: {search.best_params_}")

# Start DagsHub/MLflow run for logging
print("Logging parameters, metrics, and artifacts to DagsHub MLflow server...")
with mlflow.start_run():
    # 1. Manual logging of hyperparameters
    mlflow.log_params(search.best_params_)
    
    # Predict on validation set
    y_pred = best_model.predict(X_val)
    
    # 2. Manual logging of metrics (MAE, RMSE, R2, MAPE)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / np.clip(y_val, 1, None))) * 100
    
    print(f"Validation Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}%")
    
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MAPE", mape)

    # 3. Additional Artifact 1 — Residual Plot
    print("Generating Residual Plot artifact...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, y_val - y_pred, alpha=0.3, color='blue')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted RUL")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    residual_plot_path = "residual_plot.png"
    plt.savefig(residual_plot_path, dpi=100)
    plt.close()
    
    mlflow.log_artifact(residual_plot_path)
    print(f"Logged artifact: {residual_plot_path}")

    # 4. Additional Artifact 2 — Feature Importance Plot
    print("Generating Feature Importance Plot artifact...")
    importances = best_model.feature_importances_
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    # Keep top 15 features for clean visualization
    feat_importances = pd.Series(importances, index=X_train.columns)
    feat_importances.nlargest(15).sort_values().plot.barh(ax=ax2, color='teal')
    ax2.set_title("Top 15 Feature Importances")
    ax2.set_xlabel("Importance Value")
    feature_importance_path = "feature_importance.png"
    plt.savefig(feature_importance_path, dpi=100)
    plt.close()
    
    mlflow.log_artifact(feature_importance_path)
    print(f"Logged artifact: {feature_importance_path}")

    # 5. Log the trained model
    print("Logging model artifact to MLflow...")
    mlflow.sklearn.log_model(best_model, "model")

print("All metrics and artifacts successfully uploaded to DagsHub!")
