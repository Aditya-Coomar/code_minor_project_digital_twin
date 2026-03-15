# ==========================================================
# Koopman Neural State Space Digital Twin Training
# v4 — Clean dataset + residual decoder + detached latent
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from koopman_dataset_loader import ProcessDataset
from models.koopman_twin import KoopmanTwin


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================================
# Training Parameters
# ==========================================================

BATCH_SIZE = 64
EPOCHS = 200

LR = 1e-3

RECON_WEIGHT = 0.5
ROLLOUT_WEIGHT = 2.0
LATENT_WEIGHT = 0.3
REG_WEIGHT = 1e-4

MAX_HORIZON = 20
LATENT_CONSISTENCY_STEPS = 3

# Warmup: stay at horizon=1 for first WARMUP_EPOCHS
WARMUP_EPOCHS = 20


# ==========================================================
# Dataset
# ==========================================================

dataset = ProcessDataset()

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)

sample = dataset[0]

state_dim = sample[0].shape[-1]
control_dim = sample[1].shape[-1]

print("State dim:", state_dim)
print("Control dim:", control_dim)


# ==========================================================
# Model
# ==========================================================

model = KoopmanTwin(state_dim, control_dim).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

steps_per_epoch = len(loader)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.05,
    anneal_strategy="cos",
)

criterion = nn.MSELoss()


# ==========================================================
# Curriculum: warmup at horizon=1, then gradual ramp
# ==========================================================


def rollout_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        return 1
    return min(1 + (epoch - WARMUP_EPOCHS) // 8, MAX_HORIZON)


# ==========================================================
# Training Loop
# ==========================================================

best_r2 = -float("inf")

for epoch in range(EPOCHS):
    horizon = rollout_schedule(epoch)

    preds_all = []
    targets_all = []
    epoch_loss = 0.0
    n_batches = 0

    model.train()

    for x_hist, u_hist, u_future, y_future in loader:
        x_hist = x_hist.to(DEVICE)
        u_hist = u_hist.to(DEVICE)

        u_future = u_future[:, :horizon].to(DEVICE)
        y_future = y_future[:, :horizon].to(DEVICE)

        optimizer.zero_grad()

        # ======================================================
        # Encode initial latent state
        # ======================================================

        z = model.encoder(x_hist, u_hist)

        # ======================================================
        # Reconstruction Loss (residual decoder)
        # decode(z, x_{-2}) should reconstruct x_{-1}
        # ======================================================

        x_current = x_hist[:, -1]
        x_prev_recon = x_hist[:, -2]

        x_recon = model.decoder(z, x_prev_recon)
        recon_loss = criterion(x_recon, x_current)

        # ======================================================
        # Pure autoregressive multi-step rollout
        # Each step: x_pred = x_prev + delta(z, x_prev)
        # ======================================================

        preds_seq = []
        z_roll = z
        x_prev = x_current

        for t in range(horizon):
            u_t = u_future[:, t]
            z_roll = model.dynamics(z_roll, u_t)
            x_pred = model.decoder(z_roll, x_prev)
            preds_seq.append(x_pred)
            x_prev = x_pred

        preds_seq = torch.stack(preds_seq, dim=1)

        # ======================================================
        # Rollout loss — uniform weighting
        # ======================================================

        rollout_loss = criterion(preds_seq, y_future)

        # ======================================================
        # Multi-step latent consistency loss
        # CRITICAL: targets fully detached
        # ======================================================

        latent_loss = torch.tensor(0.0, device=DEVICE)
        n_consistency = min(LATENT_CONSISTENCY_STEPS, horizon)

        z_pred = z
        for t in range(n_consistency):
            z_pred = model.dynamics(z_pred, u_future[:, t])

            with torch.no_grad():
                shift = t + 1
                next_hist = torch.cat(
                    [x_hist[:, shift:], y_future[:, :shift]], dim=1
                )
                next_u = torch.cat(
                    [u_hist[:, shift:], u_future[:, :shift]], dim=1
                )
                z_target = model.encoder(next_hist, next_u)

            latent_loss = latent_loss + criterion(z_pred, z_target)

        latent_loss = latent_loss / n_consistency

        # ======================================================
        # Koopman operator regularization
        # ======================================================

        A = model.dynamics.A.weight
        reg_loss = torch.norm(A, p="fro")

        # ======================================================
        # Total Loss
        # ======================================================

        loss = (
            RECON_WEIGHT * recon_loss
            + ROLLOUT_WEIGHT * rollout_loss
            + LATENT_WEIGHT * latent_loss
            + REG_WEIGHT * reg_loss
        )

        # ======================================================
        # Backprop
        # ======================================================

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # ======================================================
        # Metrics
        # ======================================================

        epoch_loss += loss.item()
        n_batches += 1

        preds_all.append(preds_seq[:, -1].detach().cpu().numpy())
        targets_all.append(y_future[:, -1].cpu().numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    r2 = r2_score(targets_all, preds_all)
    avg_loss = epoch_loss / max(n_batches, 1)

    if r2 > best_r2:
        best_r2 = r2
        torch.save(model.state_dict(), "koopman_twin_best.pt")

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch + 1:3d} | Horizon {horizon:2d} | "
        f"Loss {avg_loss:.6f} | R2 {r2:.4f} | Best {best_r2:.4f} | "
        f"LR {current_lr:.2e}"
    )

print(f"\nTraining complete. Best R2: {best_r2:.4f}")
