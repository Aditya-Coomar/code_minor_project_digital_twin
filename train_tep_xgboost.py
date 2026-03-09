# ==========================================================
# Tennessee Eastman - XGBoost Digital Twin (MAX GPU VERSION)
# Standalone Script - Full GPU Utilization
# ==========================================================

import os
import gc
import datetime
import joblib
import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import xgboost as xgb

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = "data/TEP_DATASET"
N_ESTIMATORS = 800
MAX_DEPTH = 10
LEARNING_RATE = 0.05

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results/xgb_run_{RUN_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Saving results to:", OUTPUT_DIR)

# ==========================================================
# LOAD DATA
# ==========================================================

def load_rdata(file_name):
    result = pyreadr.read_r(os.path.join(DATA_PATH, file_name))
    df = list(result.values())[0]
    print(file_name, "shape:", df.shape)
    return df


train_df = load_rdata("TEP_FaultFree_Training.RData")
val_df = load_rdata("TEP_FaultFree_Testing.RData")

# ==========================================================
# PREPROCESSING
# ==========================================================

def extract_process_data(df):
    runs = df["simulationRun"].values
    process_data = df.iloc[:, 3:].values
    return runs, process_data


train_runs, train_X = extract_process_data(train_df)
val_runs, val_X = extract_process_data(val_df)

del train_df
del val_df
gc.collect()

# Scaling
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# ==========================================================
# CREATE STATIC TRAINING PAIRS
# ==========================================================

def create_static_pairs(runs, data):
    X_list, y_list = [], []
    unique_runs = np.unique(runs)

    for run in unique_runs:
        mask = runs == run
        run_data = data[mask]
        X_list.append(run_data[:-1])
        y_list.append(run_data[1:])

    return np.vstack(X_list), np.vstack(y_list)


X_train_static, y_train_static = create_static_pairs(train_runs, train_X)
X_val_static, y_val_static = create_static_pairs(val_runs, val_X)

print("Training samples:", X_train_static.shape)
print("Validation samples:", X_val_static.shape)

# ==========================================================
# MAX GPU XGBOOST CONFIGURATION
# ==========================================================

xgb_model = XGBRegressor(
    # Core
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,

    # FULL GPU
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    gpu_id=0,
    device="cuda",

    # GPU parallelism boosters
    max_bin=512,
    subsample=0.9,
    colsample_bytree=0.9,
    sampling_method="gradient_based",

    # Multi-output
    multi_strategy="multi_output_tree",

    # Regularization
    reg_lambda=1.0,
    reg_alpha=0.0,

    n_jobs=-1,
    verbosity=1
)

# ==========================================================
# TRAINING
# ==========================================================

print("\n🔥 Training XGBoost (MAX GPU)...")

xgb_model.fit(
    X_train_static,
    y_train_static,
    eval_set=[(X_val_static, y_val_static)],
    verbose=True
)

# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating model...")

val_preds = xgb_model.predict(X_val_static)

phase0_mse = mean_squared_error(y_val_static, val_preds)
phase0_mae = mean_absolute_error(y_val_static, val_preds)
phase0_r2 = r2_score(y_val_static, val_preds)

print("\nPHASE 0 RESULTS")
print("MSE:", phase0_mse)
print("MAE:", phase0_mae)
print("R2 :", phase0_r2)

# Save model
joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))

# ==========================================================
# VISUALIZATIONS
# ==========================================================

# 1️⃣ Dynamic Sensor Plot (highest variance sensor)
sensor_idx = np.argmax(np.var(y_val_static, axis=0))

plt.figure(figsize=(10, 5))
plt.plot(y_val_static[:500, sensor_idx], label="Actual")
plt.plot(val_preds[:500, sensor_idx], label="Predicted")
plt.legend()
plt.title(f"Sensor {sensor_idx} Prediction (XGBoost)")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_plot.png"))
plt.close()

# 2️⃣ Per-Sensor R2
r2_per_sensor = [
    r2_score(y_val_static[:, i], val_preds[:, i])
    for i in range(y_val_static.shape[1])
]

plt.figure(figsize=(12, 4))
plt.bar(range(len(r2_per_sensor)), r2_per_sensor)
plt.title("Per-Sensor R2 (XGBoost)")
plt.xlabel("Sensor Index")
plt.ylabel("R2")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "r2_per_sensor.png"))
plt.close()

# 3️⃣ Residual Distribution
residuals = y_val_static - val_preds

plt.figure()
plt.hist(residuals.flatten(), bins=100)
plt.title("Residual Distribution (XGBoost)")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "residual_distribution.png"))
plt.close()

# 4️⃣ Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=20)
plt.title("Top 20 Feature Importances")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()

# ==========================================================
# SAVE METRICS
# ==========================================================

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write("PHASE 0 - XGBOOST RESULTS\n")
    f.write(f"MSE: {phase0_mse}\n")
    f.write(f"MAE: {phase0_mae}\n")
    f.write(f"R2 : {phase0_r2}\n")

print("\nAll artifacts saved successfully.")