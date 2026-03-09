# ==========================================================
# DIGITAL TWIN TRAINING PIPELINE
# Adaptive Multi-Step Multi-Head LSTM
# Stable Training Version
# ==========================================================

import os
import gc
import datetime
import numpy as np
import pyreadr
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt

from multiprocessing import freeze_support
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "data/TEP_DATASET"

SEQ_LEN = 180
PRED_STEPS = 3

BATCH_SIZE = 192
HIDDEN_DIM = 256
NUM_LAYERS = 2

EPOCHS = 120
LR = 0.001

SMOOTHNESS_WEIGHT = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.n_xmv = input_dim-n_xmeas

        self.lstm = nn.LSTM(
            input_dim,
            HIDDEN_DIM,
            NUM_LAYERS,
            batch_first=True,
            dropout=0.2
        )

        self.attn = TemporalAttention(HIDDEN_DIM)
        self.norm = nn.LayerNorm(HIDDEN_DIM)

        self.xmeas_head = nn.Linear(HIDDEN_DIM,n_xmeas*PRED_STEPS)
        self.xmv_head = nn.Linear(HIDDEN_DIM,self.n_xmv*PRED_STEPS)

    def forward(self,x):

        lstm_out,_ = self.lstm(x)

        context = self.attn(lstm_out)
        context = self.norm(context)

        xmeas = self.xmeas_head(context)
        xmv = self.xmv_head(context)

        batch = x.shape[0]

        xmeas = xmeas.view(batch,PRED_STEPS,self.n_xmeas)
        xmv = xmv.view(batch,PRED_STEPS,self.n_xmv)

        return torch.cat([xmeas,xmv],dim=2)

# ==========================================================
# DATASET
# ==========================================================

class TEPTwinDataset(Dataset):

    def __init__(self,runs,data,seq_len,pred_steps):

        self.data=data
        self.seq_len=seq_len
        self.pred_steps=pred_steps

        self.indices=[]

        unique_runs=np.unique(runs)

        for run in unique_runs:

            idx=np.where(runs==run)[0]

            for i in range(len(idx)-seq_len-pred_steps):

                self.indices.append(idx[i])

        self.indices=np.array(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,idx):

        start=self.indices[idx]

        seq=self.data[start:start+self.seq_len]

        last=seq[-1]

        future=self.data[
            start+self.seq_len:start+self.seq_len+self.pred_steps
        ]

        residual=future-last

        return (
            torch.tensor(seq,dtype=torch.float32),
            torch.tensor(residual,dtype=torch.float32),
            torch.tensor(last,dtype=torch.float32)
        )

# ==========================================================
# LOAD DATA
# ==========================================================

def load_rdata(file):

    result = pyreadr.read_r(os.path.join(DATA_PATH,file))
    return list(result.values())[0]

# ==========================================================
# TRAINING
# ==========================================================

def main():

    print("Device:",DEVICE)

    if DEVICE.type=="cuda":
        print("GPU:",torch.cuda.get_device_name(0))

    RUN_ID=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR=f"results/run_{RUN_ID}"

    os.makedirs(OUTPUT_DIR,exist_ok=True)

    print("Loading dataset...")

    train_df=load_rdata("TEP_FaultFree_Training.RData")
    val_df=load_rdata("TEP_FaultFree_Testing.RData")

    sensor_names=train_df.columns[3:].tolist()

    xmeas_cols=[c for c in sensor_names if "xmeas" in c]
    n_xmeas=len(xmeas_cols)

    train_runs=train_df["simulationRun"].values
    val_runs=val_df["simulationRun"].values

    train_X=train_df.iloc[:,3:].values
    val_X=val_df.iloc[:,3:].values

    del train_df,val_df
    gc.collect()

    print("Normalizing...")

    scaler=StandardScaler()

    train_X=scaler.fit_transform(train_X)
    val_X=scaler.transform(val_X)

    joblib.dump(scaler,os.path.join(OUTPUT_DIR,"scaler.pkl"))

    train_dataset=TEPTwinDataset(train_runs,train_X,SEQ_LEN,PRED_STEPS)
    val_dataset=TEPTwinDataset(val_runs,val_X,SEQ_LEN,PRED_STEPS)

    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model=MultiHeadTwin(train_X.shape[1],n_xmeas).to(DEVICE)

    optimizer=optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode="min",factor=0.5,patience=5
    )

    scaler_amp=GradScaler()

    # sensor weights
    sensor_var=np.var(train_X,axis=0)
    sensor_weights=1/(sensor_var+1e-6)
    sensor_weights=sensor_weights/sensor_weights.mean()
    sensor_weights=torch.tensor(sensor_weights).to(DEVICE)

    step_weights=torch.tensor([1.0,0.8,0.5],device=DEVICE)

    huber = nn.SmoothL1Loss(reduction="none", beta=0.5)

    train_losses=[]

    print("\nTraining...\n")

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss=0

        for xb,residuals,last_state in train_loader:

            xb=xb.to(DEVICE,non_blocking=True)
            residuals=residuals.to(DEVICE,non_blocking=True)
            last_state=last_state.to(DEVICE,non_blocking=True)

            optimizer.zero_grad()

            with autocast():

                pred_res=model(xb)

                true_next=last_state.unsqueeze(1)+residuals
                pred_next=last_state.unsqueeze(1)+pred_res

                weights=sensor_weights.unsqueeze(0).unsqueeze(0)

                error=huber(pred_next,true_next)

                error=error*weights
                error=error*step_weights.view(1,PRED_STEPS,1)

                weighted_loss=error.mean()

                smooth=torch.mean(torch.abs(pred_next[:,1:]-pred_next[:,:-1]))

                loss=weighted_loss+SMOOTHNESS_WEIGHT*smooth

            scaler_amp.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss+=loss.item()

        avg_loss=epoch_loss/len(train_loader)

        scheduler.step(avg_loss)

        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {avg_loss:.4f}")

    print("Saving model...")

    torch.save(model.state_dict(),os.path.join(OUTPUT_DIR,"twin_model.pt"))

    # ======================================================
    # TRAINING LOSS PLOT
    # ======================================================

    plt.figure(figsize=(8,5))
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.savefig(os.path.join(OUTPUT_DIR,"training_loss.png"))
    plt.close()

    # ======================================================
    # EVALUATION
    # ======================================================

    model.eval()

    preds=[]
    actuals=[]

    with torch.no_grad():

        for xb,residuals,last_state in val_loader:

            xb=xb.to(DEVICE)

            pred_res=model(xb)

            pred_next=last_state.unsqueeze(1).to(DEVICE)+pred_res
            true_next=last_state.unsqueeze(1).to(DEVICE)+residuals.to(DEVICE)

            preds.append(pred_next[:,0].cpu().numpy())
            actuals.append(true_next[:,0].cpu().numpy())

    val_preds=np.vstack(preds)
    val_actuals=np.vstack(actuals)

    mse=mean_squared_error(val_actuals,val_preds)
    mae=mean_absolute_error(val_actuals,val_preds)
    r2=r2_score(val_actuals,val_preds)

    print("\nFINAL RESULTS")
    print("MSE:",mse)
    print("MAE:",mae)
    print("R2 :",r2)

    # ======================================================
    # RESIDUAL DISTRIBUTION
    # ======================================================

    residuals = val_actuals-val_preds

    plt.figure(figsize=(8,5))
    plt.hist(residuals.flatten(),bins=120)

    plt.title("Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.savefig(os.path.join(OUTPUT_DIR,"residual_distribution.png"))
    plt.close()

    print("\nArtifacts saved to:",OUTPUT_DIR)

# ==========================================================

if __name__=="__main__":

    freeze_support()

    main()