# ==========================================================
# Per-Sensor Analysis Script (NO RETRAINING)
# Loads saved model + scaler
# Generates per-sensor plots with R²
# ==========================================================

import os
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/TEP_DATASET"
RUN_FOLDER = "results/run_20260308_040819"  # <<< CHANGE THIS

SEQ_LEN = 120  # must match training config
HIDDEN_DIM = 192  # must match training config
NUM_LAYERS = 2
BATCH_SIZE = 96

DEVICE = torch.device("cuda")

# ==========================================================
# LOAD SCALER
# ==========================================================

scaler = joblib.load(os.path.join(RUN_FOLDER, "scaler.pkl"))

# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================


def load_rdata(file_name):
    result = pyreadr.read_r(os.path.join(DATA_PATH, file_name))
    df = list(result.values())[0]
    return df


val_df = load_rdata("TEP_FaultFree_Testing.RData")

sensor_names = val_df.columns[3:].tolist()

val_runs = val_df["simulationRun"].values
val_X = val_df.iloc[:, 3:].values

val_X = scaler.transform(val_X)

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


val_dataset = RunAwareSequenceDataset(val_runs, val_X, SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================================
# MODEL DEFINITION (MUST MATCH TRAINING)
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


model = LSTMTwin(input_dim=val_X.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(RUN_FOLDER, "lstm_model.pt")))
model.eval()

# ==========================================================
# INFERENCE
# ==========================================================

preds_list = []
actuals_list = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds_list.append(out.cpu().numpy())
        actuals_list.append(yb.numpy())

val_preds = np.vstack(preds_list)
val_actuals = np.vstack(actuals_list)

# ==========================================================
# PER-SENSOR PLOTS
# ==========================================================

sensor_plot_dir = os.path.join(RUN_FOLDER, "per_sensor_plots")
os.makedirs(sensor_plot_dir, exist_ok=True)

r2_values = []

for i, sensor_name in enumerate(sensor_names):
    r2_val = r2_score(val_actuals[:, i], val_preds[:, i])
    r2_values.append(r2_val)

    plt.figure(figsize=(10, 4))
    plt.plot(val_actuals[:500, i], label="Actual", linewidth=1)
    plt.plot(val_preds[:500, i], label="Predicted", linewidth=1)
    plt.legend()
    plt.title(f"{sensor_name} | R2 = {r2_val:.4f}")
    plt.xlabel("Time Step")
    plt.ylabel("Scaled Value")
    plt.grid(True)

    filename = f"{sensor_name}_R2_{r2_val:.3f}.png"
    plt.savefig(os.path.join(sensor_plot_dir, filename))
    plt.close()

# ==========================================================
# SAVE R2 TABLE
# ==========================================================

sensor_r2_df = pd.DataFrame({"Sensor": sensor_names, "R2": r2_values})

sensor_r2_df.sort_values("R2", ascending=False).to_csv(
    os.path.join(RUN_FOLDER, "sensor_r2_values.csv"), index=False
)

print("Per-sensor analysis completed.")
