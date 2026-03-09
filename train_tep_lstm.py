# ==========================================================
# Tennessee Eastman Digital Twin
# Temporal Attention + Residual LSTM
# GPU Optimized / Windows Safe
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
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/TEP_DATASET"

SEQ_LEN = 160
BATCH_SIZE = 192
HIDDEN_DIM = 256
NUM_LAYERS = 2
EPOCHS = 120
LR = 0.001

# ==========================================================
# DATASET
# ==========================================================

class RunAwareResidualDataset(Dataset):

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

        seq = self.data[start:start + self.seq_len]

        next_state = self.data[start + self.seq_len]

        last_state = seq[-1]

        residual = next_state - last_state

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(residual, dtype=torch.float32),
            torch.tensor(last_state, dtype=torch.float32),
        )

# ==========================================================
# TEMPORAL ATTENTION
# ==========================================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):

        # lstm_outputs shape:
        # (batch, seq_len, hidden)

        weights = torch.softmax(
            self.attn(lstm_outputs),
            dim=1
        )

        context = torch.sum(weights * lstm_outputs, dim=1)

        return context

# ==========================================================
# DIGITAL TWIN MODEL
# ==========================================================

class AttentionResidualTwin(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.2
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
# MAIN
# ==========================================================

def main():

    DEVICE = torch.device("cuda")

    print("Using GPU:", torch.cuda.get_device_name(0))

    # ======================================================
    # OUTPUT DIR
    # ======================================================

    RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    OUTPUT_DIR = f"results/run_{RUN_ID}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving results to:", OUTPUT_DIR)

    # ======================================================
    # LOAD DATA
    # ======================================================

    def load_rdata(file_name):

        result = pyreadr.read_r(os.path.join(DATA_PATH, file_name))

        df = list(result.values())[0]

        print(file_name, df.shape)

        return df

    train_df = load_rdata("TEP_FaultFree_Training.RData")

    val_df = load_rdata("TEP_FaultFree_Testing.RData")

    sensor_names = train_df.columns[3:].tolist()

    # ======================================================
    # EXTRACT DATA
    # ======================================================

    def extract(df):

        runs = df["simulationRun"].values

        X = df.iloc[:, 3:].values

        return runs, X

    train_runs, train_X = extract(train_df)
    val_runs, val_X = extract(val_df)

    del train_df
    del val_df

    gc.collect()

    # ======================================================
    # NORMALIZATION
    # ======================================================

    scaler = StandardScaler()

    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    # ======================================================
    # DATASETS
    # ======================================================

    train_dataset = RunAwareResidualDataset(train_runs, train_X, SEQ_LEN)

    val_dataset = RunAwareResidualDataset(val_runs, val_X, SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ======================================================
    # MODEL
    # ======================================================

    model = AttentionResidualTwin(train_X.shape[1]).to(DEVICE)

    criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    scaler_amp = GradScaler()

    # ======================================================
    # TRAINING
    # ======================================================

    train_losses = []

    print("\nTraining Attention Residual Twin\n")

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        for xb, residual, last_state in train_loader:

            xb = xb.to(DEVICE, non_blocking=True)
            residual = residual.to(DEVICE, non_blocking=True)
            last_state = last_state.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast():

                pred_residual = model(xb)

                pred_next = last_state + pred_residual

                true_next = last_state + residual

                loss = criterion(pred_next, true_next)

            scaler_amp.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        scheduler.step(avg_loss)

        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")

    # ======================================================
    # SAVE MODEL
    # ======================================================

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_DIR, "lstm_model.pt")
    )

    # ======================================================
    # EVALUATION
    # ======================================================

    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():

        for xb, residual, last_state in val_loader:

            xb = xb.to(DEVICE)

            residual = residual.to(DEVICE)

            last_state = last_state.to(DEVICE)

            pred_residual = model(xb)

            pred_next = last_state + pred_residual

            true_next = last_state + residual

            preds.append(pred_next.cpu().numpy())

            actuals.append(true_next.cpu().numpy())

    val_preds = np.vstack(preds)

    val_actuals = np.vstack(actuals)

    mse = mean_squared_error(val_actuals, val_preds)
    mae = mean_absolute_error(val_actuals, val_preds)
    r2 = r2_score(val_actuals, val_preds)

    print("\nFINAL RESULTS")

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 :", r2)

    # ======================================================
    # PLOTS
    # ======================================================

    plt.figure()

    plt.plot(train_losses)

    plt.title("Training Loss")

    plt.grid(True)

    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))

    plt.close()

    residuals = val_actuals - val_preds

    plt.figure()

    plt.hist(residuals.flatten(), bins=100)

    plt.title("Residual Distribution")

    plt.savefig(os.path.join(OUTPUT_DIR, "residual_distribution.png"))

    plt.close()

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:

        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R2 : {r2}\n")

    with open(os.path.join(OUTPUT_DIR, "config.txt"), "w") as f:
        f.write(f"SEQ_LEN={SEQ_LEN}\n")
        f.write(f"HIDDEN_DIM={HIDDEN_DIM}\n")
        f.write(f"NUM_LAYERS={NUM_LAYERS}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"EPOCHS={EPOCHS}\n")

    print("\nArtifacts saved.")

# ==========================================================
# WINDOWS SAFE ENTRY
# ==========================================================

if __name__ == "__main__":

    main()