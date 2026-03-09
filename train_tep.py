# ==========================================================
# Tennessee Eastman Digital Twin (Local GTX1050 Version)
# Run-aware, Memory-safe, Optimized
# ==========================================================

import os
import gc
import datetime
import joblib
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = "data/TEP_DATASET"

SEQ_LEN = 80
BATCH_SIZE = 32
HIDDEN_DIM = 256
NUM_LAYERS = 2
EPOCHS = 120
LR = 0.001

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("Using GPU:", torch.cuda.get_device_name(0))

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results/run_{RUN_ID}"
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

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# ==========================================================
# PHASE 0 - XGBOOST STATIC BASELINE (GPU)
# ==========================================================

"""
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

xgb_model = XGBRegressor(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    device="cuda",
    multi_strategy="multi_output_tree",
)

print("Training Phase 0 (XGBoost)...")
xgb_model.fit(X_train_static, y_train_static)

val_preds_xgb = xgb_model.predict(X_val_static)

phase0_mse = mean_squared_error(y_val_static, val_preds_xgb)
phase0_mae = mean_absolute_error(y_val_static, val_preds_xgb)
phase0_r2 = r2_score(y_val_static, val_preds_xgb)

print("\nPHASE 0 RESULTS")
print("MSE:", phase0_mse)
print("MAE:", phase0_mae)
print("R2 :", phase0_r2)

joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
"""
# ==========================================================
# RUN-AWARE DATASET
# ==========================================================


class RunAwareSequenceDataset(Dataset):
    def __init__(self, runs, data, seq_len):
        self.seq_len = seq_len
        self.data = data
        self.indices = []

        unique_runs = np.unique(runs)

        for run in unique_runs:
            run_indices = np.where(runs == run)[0]
            for i in range(len(run_indices) - seq_len):
                self.indices.append(run_indices[i])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


train_dataset = RunAwareSequenceDataset(train_runs, train_X, SEQ_LEN)
val_dataset = RunAwareSequenceDataset(val_runs, val_X, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================================
# PHASE 1 - LSTM DIGITAL TWIN
# ==========================================================


class LSTMTwin(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True, dropout=0.2
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.fc(out)


model = LSTMTwin(input_dim=train_X.shape[1]).to(DEVICE)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

scaler_amp = GradScaler()

# ==========================================================
# TRAINING LOOP
# ==========================================================

print("\nTraining Phase 1 (LSTM)...")

train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            preds = model(xb)
            loss = criterion(preds, yb)

        scaler_amp.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_model.pt"))

# ==========================================================
# EVALUATION
# ==========================================================

model.eval()
preds_list, actuals_list = [], []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds_list.append(out.cpu().numpy())
        actuals_list.append(yb.numpy())

val_preds = np.vstack(preds_list)
val_actuals = np.vstack(actuals_list)

phase1_mse = mean_squared_error(val_actuals, val_preds)
phase1_mae = mean_absolute_error(val_actuals, val_preds)
phase1_r2 = r2_score(val_actuals, val_preds)

print("\nPHASE 1 RESULTS")
print("MSE:", phase1_mse)
print("MAE:", phase1_mae)
print("R2 :", phase1_r2)

# ==========================================================
# SAVE VISUALIZATIONS
# ==========================================================

# Training loss
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
plt.close()

# Dynamic sensor plot
sensor_idx = np.argmax(np.var(val_actuals, axis=0))

plt.figure(figsize=(10, 5))
plt.plot(val_actuals[:500, sensor_idx], label="Actual")
plt.plot(val_preds[:500, sensor_idx], label="Predicted")
plt.legend()
plt.title(f"Sensor {sensor_idx} Prediction")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_plot.png"))
plt.close()

# Per-sensor R2
r2_per_sensor = [
    r2_score(val_actuals[:, i], val_preds[:, i]) for i in range(val_actuals.shape[1])
]

plt.figure(figsize=(12, 4))
plt.bar(range(len(r2_per_sensor)), r2_per_sensor)
plt.title("Per-Sensor R2")
plt.xlabel("Sensor Index")
plt.ylabel("R2")
plt.savefig(os.path.join(OUTPUT_DIR, "r2_per_sensor.png"))
plt.close()

# Residual distribution
residuals = val_actuals - val_preds

plt.figure()
plt.hist(residuals.flatten(), bins=100)
plt.title("Residual Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "residual_distribution.png"))
plt.close()

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write("PHASE 0 RESULTS\n")
    f.write(f"MSE: {'NA'}\n")
    f.write(f"MAE: {'NA'}\n")
    f.write(f"R2 : {'NA'}\n\n")

    f.write("PHASE 1 RESULTS\n")
    f.write(f"MSE: {phase1_mse}\n")
    f.write(f"MAE: {phase1_mae}\n")
    f.write(f"R2 : {phase1_r2}\n")

# Save config
with open(os.path.join(OUTPUT_DIR, "config.txt"), "w") as f:
    f.write(f"SEQ_LEN={SEQ_LEN}\n")
    f.write(f"HIDDEN_DIM={HIDDEN_DIM}\n")
    f.write(f"NUM_LAYERS={NUM_LAYERS}\n")
    f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
    f.write(f"EPOCHS={EPOCHS}\n")

print("\nAll artifacts saved successfully.")
