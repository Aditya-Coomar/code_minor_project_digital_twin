import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch.utils.data import DataLoader

from dataset_stream import TEPDataset
from models.state_space_twin import StateSpaceTwin


def main():

    torch.backends.cudnn.benchmark = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    BATCH_SIZE = 512
    EPOCHS = 50
    LR = 1e-3

    dataset = TEPDataset("X_seq.npy", "U_seq.npy", "y.npy")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # stable for Windows + memmap
        pin_memory=True,
    )

    model = StateSpaceTwin(state_dim=41, control_dim=11).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss = 0
        preds_all = []
        targets_all = []

        for X_batch, U_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            U_batch = U_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                preds = model(X_batch, U_batch)

                loss = criterion(preds, y_batch)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(y_batch.detach().cpu().numpy())

        avg_loss = epoch_loss / len(loader)

        preds_all = np.vstack(preds_all)
        targets_all = np.vstack(targets_all)

        mse = mean_squared_error(targets_all, preds_all)
        mae = mean_absolute_error(targets_all, preds_all)
        r2 = r2_score(targets_all, preds_all)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss {avg_loss:.6f} | "
            f"MSE {mse:.6f} | "
            f"MAE {mae:.6f} | "
            f"R2 {r2:.6f}"
        )

    torch.save(model.state_dict(), "state_space_twin.pt")

    print("\nModel saved")


if __name__ == "__main__":
    main()
