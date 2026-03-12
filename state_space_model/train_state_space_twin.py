import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from sklearn.metrics import r2_score, mean_absolute_error

from torch.utils.data import DataLoader

from dataset_stream import TEPDataset
from models.state_space_twin import StateSpaceTwin


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
EPOCHS = 80
LR = 3e-4


def rollout_schedule(epoch):

    if epoch < 10:
        return 5
    elif epoch < 30:
        return 10
    else:
        return 20


def train():

    dataset = TEPDataset("X_seq.npy", "U_seq.npy", "y.npy")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )

    model = StateSpaceTwin(state_dim=41, control_dim=11, hidden_dim=384).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = nn.MSELoss()

    scaler = torch.amp.GradScaler()

    results = []

    for epoch in range(EPOCHS):
        model.train()

        horizon = rollout_schedule(epoch)

        preds = []
        targets = []

        epoch_loss = 0

        for X, U, y in loader:
            X = X.to(DEVICE)
            U = U.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = model(X, U, horizon)

                loss = criterion(pred, y)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            preds.append(pred.detach().cpu().numpy())
            targets.append(y.cpu().numpy())

        scheduler.step()

        preds = np.vstack(preds)
        targets = np.vstack(targets)

        mse = ((preds - targets) ** 2).mean()
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)

        avg_loss = epoch_loss / len(loader)

        print(
            f"Epoch {epoch + 1:03d} | Horizon {horizon} | "
            f"Loss {avg_loss:.4f} | MSE {mse:.4f} | MAE {mae:.4f} | R2 {r2:.4f}"
        )

        results.append(
            {
                "epoch": epoch + 1,
                "horizon": horizon,
                "loss": avg_loss,
                "mse": mse,
                "mae": mae,
                "r2": r2,
            }
        )

    torch.save(model.state_dict(), "state_space_twin.pt")

    pd.DataFrame(results).to_csv("training_results.csv", index=False)

    print("Training finished")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    train()
