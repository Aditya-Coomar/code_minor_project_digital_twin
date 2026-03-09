import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from dataset_stream import TEPDataset
from models.state_space_twin import StateSpaceTwin


def rollout_horizon(epoch):

    if epoch < 5:
        return 1
    elif epoch < 15:
        return 3
    elif epoch < 30:
        return 5
    else:
        return 8


def main():

    torch.backends.cudnn.benchmark = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    BATCH_SIZE = 512
    EPOCHS = 60
    LR = 1e-3

    dataset = TEPDataset("X_seq.npy", "U_seq.npy", "y.npy")

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )

    model = StateSpaceTwin(state_dim=41, control_dim=11, hidden_dim=256).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # store results
    results = []

    for epoch in range(EPOCHS):
        model.train()

        horizon = rollout_horizon(epoch)

        epoch_loss = 0

        preds_all = []
        targets_all = []

        for X_batch, U_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            U_batch = U_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                h = model.encoder(X_batch, U_batch)

                x_current = X_batch[:, -1, :]

                loss = 0

                for k in range(horizon):
                    u = U_batch[:, -1, :]

                    h = model.dynamics(h, u)

                    delta = model.head(h)

                    x_pred = x_current + delta

                    loss += criterion(x_pred, y_batch)

                    x_current = x_pred

                loss = loss / horizon

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)

            scaler.update()

            epoch_loss += loss.item()

            preds_all.append(x_pred.detach().cpu().numpy())
            targets_all.append(y_batch.detach().cpu().numpy())

        scheduler.step()

        avg_loss = epoch_loss / len(loader)

        preds_all = np.vstack(preds_all)
        targets_all = np.vstack(targets_all)

        mse = mean_squared_error(targets_all, preds_all)
        mae = mean_absolute_error(targets_all, preds_all)
        r2 = r2_score(targets_all, preds_all)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Horizon {horizon} | "
            f"Loss {avg_loss:.6f} | "
            f"MSE {mse:.6f} | "
            f"MAE {mae:.6f} | "
            f"R2 {r2:.6f}"
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

    # Save model
    torch.save(model.state_dict(), "state_space_twin_rollout.pt")

    print("\nModel saved")

    # Save results CSV
    df = pd.DataFrame(results)

    df.to_csv("training_results.csv", index=False)

    print("Saved training_results.csv")

    # Save JSON
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Saved training_results.json")

    # Plot training curves
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["loss"])
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["r2"])
    plt.title("R² Score")

    plt.tight_layout()
    plt.savefig("training_curve.png")

    print("Saved training_curve.png")


if __name__ == "__main__":
    main()
