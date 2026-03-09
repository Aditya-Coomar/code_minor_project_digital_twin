# ==========================================================
# FAST DIGITAL TWIN ANALYSIS SCRIPT
# Attention + Residual LSTM
# Windows multiprocessing safe
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

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/TEP_DATASET"
RUN_FOLDER = "results/run_20260309_030639"  # CHANGE THIS

SEQ_LEN = 160
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 192

DEVICE = torch.device("cuda")

# ==========================================================
# DATASET
# ==========================================================


class RunAwareDataset(Dataset):
    def __init__(self, runs, data, seq_len):

        self.seq_len = seq_len
        self.data = data
        self.indices = []

        unique_runs = np.unique(runs)

        for run in unique_runs:
            run_idx = np.where(runs == run)[0]

            for i in range(len(run_idx) - seq_len):
                self.indices.append(run_idx[i])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        start = self.indices[idx]

        x = self.data[start : start + self.seq_len]

        y = self.data[start + self.seq_len]

        last_state = x[-1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(last_state, dtype=torch.float32),
        )


# ==========================================================
# MODEL
# ==========================================================


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):

        weights = torch.softmax(self.attn(lstm_out), dim=1)

        context = torch.sum(weights * lstm_out, dim=1)

        return context


class AttentionResidualTwin(nn.Module):
    def __init__(self, input_dim):

        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.2,
        )

        self.attn = TemporalAttention(HIDDEN_DIM)

        self.norm = nn.LayerNorm(HIDDEN_DIM)

        self.fc = nn.Linear(HIDDEN_DIM, input_dim)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        context = self.attn(lstm_out)

        context = self.norm(context)

        residual = self.fc(context)

        return residual


# ==========================================================
# DATA LOADING
# ==========================================================


def load_rdata(file_name):

    result = pyreadr.read_r(os.path.join(DATA_PATH, file_name))

    df = list(result.values())[0]

    return df


# ==========================================================
# MAIN EXECUTION
# ==========================================================


def main():

    print("Loading scaler...")

    scaler = joblib.load(os.path.join(RUN_FOLDER, "scaler.pkl"))

    print("Loading dataset...")

    val_df = load_rdata("TEP_FaultFree_Testing.RData")

    sensor_names = val_df.columns[3:].tolist()

    val_runs = val_df["simulationRun"].values
    val_X = val_df.iloc[:, 3:].values

    val_X = scaler.transform(val_X)

    # dataset
    val_dataset = RunAwareDataset(val_runs, val_X, SEQ_LEN)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("Loading model...")

    model = AttentionResidualTwin(val_X.shape[1]).to(DEVICE)

    model.load_state_dict(
        torch.load(os.path.join(RUN_FOLDER, "lstm_model.pt"), map_location=DEVICE)
    )

    model.eval()

    print("Model loaded successfully")

    # ======================================================
    # INFERENCE
    # ======================================================

    preds = []
    actuals = []

    with torch.no_grad():
        for xb, yb, last_state in val_loader:
            xb = xb.to(DEVICE)
            last_state = last_state.to(DEVICE)

            residual_pred = model(xb)

            pred_next = last_state + residual_pred

            preds.append(pred_next.cpu().numpy())
            actuals.append(yb.numpy())

    val_preds = np.vstack(preds)
    val_actuals = np.vstack(actuals)

    print("Inference completed")

    # ======================================================
    # R2 PER SENSOR
    # ======================================================

    sensor_r2 = []

    for i in range(len(sensor_names)):
        r2_val = r2_score(val_actuals[:, i], val_preds[:, i])

        sensor_r2.append(r2_val)

    sensor_r2_df = pd.DataFrame({"Sensor": sensor_names, "R2": sensor_r2}).sort_values(
        "R2", ascending=False
    )

    sensor_r2_df.to_csv(os.path.join(RUN_FOLDER, "sensor_r2_values.csv"), index=False)

    print("R² table saved")

    # ======================================================
    # PLOTS
    # ======================================================

    plot_dir = os.path.join(RUN_FOLDER, "per_sensor_plots")

    os.makedirs(plot_dir, exist_ok=True)

    for i, sensor in enumerate(sensor_names):
        r2_val = sensor_r2[i]

        plt.figure(figsize=(10, 4))

        plt.plot(val_actuals[:500, i], label="Actual")
        plt.plot(val_preds[:500, i], label="Predicted")

        plt.title(f"{sensor} | R2 = {r2_val:.4f}")

        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(plot_dir, f"{sensor}_R2_{r2_val:.3f}.png"))

        plt.close()

    print("Per-sensor plots generated")
    print("Analysis completed successfully")


# ==========================================================
# WINDOWS SAFE ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
