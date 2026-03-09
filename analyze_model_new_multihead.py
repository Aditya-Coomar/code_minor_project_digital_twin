# ==========================================================
# DIGITAL TWIN ANALYSIS SCRIPT
# Multi-step + Multi-head Twin
# Compatible with new training architecture
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
RUN_FOLDER = "results/run_20260309_030639"   # CHANGE

SEQ_LEN = 220
PRED_STEPS = 3
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 192

DEVICE = torch.device("cuda")

# ==========================================================
# DATASET
# ==========================================================

class MultiStepDataset(Dataset):

    def __init__(self, runs, data, seq_len):

        self.seq_len = seq_len
        self.data = data
        self.indices = []

        unique_runs = np.unique(runs)

        for run in unique_runs:

            run_idx = np.where(runs == run)[0]

            for i in range(len(run_idx) - seq_len - PRED_STEPS):

                self.indices.append(run_idx[i])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        start = self.indices[idx]

        seq = self.data[start:start+SEQ_LEN]

        last_state = seq[-1]

        future = self.data[start+SEQ_LEN]

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(future, dtype=torch.float32),
            torch.tensor(last_state, dtype=torch.float32),
        )

# ==========================================================
# MODEL
# ==========================================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim,1)

    def forward(self,x):

        weights = torch.softmax(self.attn(x),dim=1)

        context = torch.sum(weights*x,dim=1)

        return context


class MultiHeadTwin(nn.Module):

    def __init__(self,input_dim,n_xmeas):

        super().__init__()

        self.n_xmeas = n_xmeas
        self.n_xmv = input_dim - n_xmeas

        self.lstm = nn.LSTM(
            input_dim,
            HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.2
        )

        self.attn = TemporalAttention(HIDDEN_DIM)

        self.norm = nn.LayerNorm(HIDDEN_DIM)

        self.xmeas_head = nn.Linear(HIDDEN_DIM, n_xmeas * PRED_STEPS)

        self.xmv_head = nn.Linear(HIDDEN_DIM, self.n_xmv * PRED_STEPS)

    def forward(self,x):

        lstm_out,_ = self.lstm(x)

        context = self.attn(lstm_out)

        context = self.norm(context)

        xmeas = self.xmeas_head(context)
        xmv = self.xmv_head(context)

        batch = x.shape[0]

        xmeas = xmeas.view(batch,PRED_STEPS,self.n_xmeas)
        xmv = xmv.view(batch,PRED_STEPS,self.n_xmv)

        pred = torch.cat([xmeas,xmv],dim=2)

        return pred

# ==========================================================
# DATA LOADER
# ==========================================================

def load_rdata(file):

    result = pyreadr.read_r(os.path.join(DATA_PATH,file))

    df = list(result.values())[0]

    return df

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("Loading scaler...")

    scaler = joblib.load(os.path.join(RUN_FOLDER,"scaler.pkl"))

    print("Loading dataset...")

    val_df = load_rdata("TEP_FaultFree_Testing.RData")

    sensor_names = val_df.columns[3:].tolist()

    xmeas_cols = [c for c in sensor_names if "xmeas" in c]

    n_xmeas = len(xmeas_cols)

    val_runs = val_df["simulationRun"].values
    val_X = val_df.iloc[:,3:].values

    val_X = scaler.transform(val_X)

    val_dataset = MultiStepDataset(val_runs,val_X,SEQ_LEN)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Loading model...")

    model = MultiHeadTwin(val_X.shape[1],n_xmeas).to(DEVICE)

    model.load_state_dict(
        torch.load(os.path.join(RUN_FOLDER,"twin_model.pt"),map_location=DEVICE)
    )

    model.eval()

    print("Model loaded successfully")

    preds=[]
    actuals=[]

    with torch.no_grad():

        for xb,yb,last_state in val_loader:

            xb = xb.to(DEVICE)

            pred_residual = model(xb)

            pred_next = last_state.unsqueeze(1).to(DEVICE) + pred_residual

            preds.append(pred_next[:,0].cpu().numpy())

            actuals.append(yb.numpy())

    val_preds = np.vstack(preds)
    val_actuals = np.vstack(actuals)

    print("Inference completed")

    # ======================================================
    # SENSOR R2
    # ======================================================

    sensor_r2=[]

    for i,name in enumerate(sensor_names):

        r2_val = r2_score(val_actuals[:,i],val_preds[:,i])

        sensor_r2.append(r2_val)

    df = pd.DataFrame({"Sensor":sensor_names,"R2":sensor_r2})

    df=df.sort_values("R2",ascending=False)

    df.to_csv(os.path.join(RUN_FOLDER,"sensor_r2_values.csv"),index=False)

    print("Sensor R2 table saved")

    # ======================================================
    # PLOTS
    # ======================================================

    plot_dir=os.path.join(RUN_FOLDER,"per_sensor_plots")

    os.makedirs(plot_dir,exist_ok=True)

    for i,sensor in enumerate(sensor_names):

        r2_val=sensor_r2[i]

        plt.figure(figsize=(10,4))

        plt.plot(val_actuals[:500,i],label="Actual")
        plt.plot(val_preds[:500,i],label="Predicted")

        plt.title(f"{sensor} | R2={r2_val:.4f}")

        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(plot_dir,f"{sensor}_R2_{r2_val:.3f}.png"))

        plt.close()

    print("Plots generated")
    print("Analysis completed")

# ==========================================================

if __name__ == "__main__":
    main()