# ==========================================================
# Advanced Diagnostic Suite for Koopman Digital Twin
# Saves all plots + metrics using real sensor names
# ==========================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from koopman_dataset_loader import ProcessDataset
from models.koopman_twin import KoopmanTwin


# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "koopman_twin_best.pt"
OUTPUT_DIR = "diagnostics"

BATCH_SIZE = 64
MAX_ROLLOUT = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Sensor names
# ==========================================================

SENSOR_NAMES = [f"xmeas_{i}" for i in range(1, 42)]
CONTROL_NAMES = [f"xmv_{i}" for i in range(1, 12)]


# ==========================================================
# Create directories
# ==========================================================

dirs = [
    "prediction_scatter",
    "prediction_timeseries",
    "residual_histograms",
    "residual_autocorr",
    "rollout",
    "metrics",
    "latent_analysis",
]

for d in dirs:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)


# ==========================================================
# Dataset
# ==========================================================

dataset = ProcessDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

sample = dataset[0]

state_dim = sample[0].shape[-1]
control_dim = sample[1].shape[-1]

print("State dim:", state_dim)
print("Control dim:", control_dim)


# ==========================================================
# Load model
# ==========================================================

model = KoopmanTwin(state_dim, control_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded")


# ==========================================================
# Prediction pass
# ==========================================================

all_preds = []
all_targets = []

with torch.no_grad():
    for x_hist, u_hist, u_future, y_future in loader:
        x_hist = x_hist.to(DEVICE)
        u_hist = u_hist.to(DEVICE)

        u_future = u_future[:, :1].to(DEVICE)
        y_future = y_future[:, :1].to(DEVICE)

        preds = model.rollout(x_hist, u_hist, u_future)

        preds = preds[:, -1]

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_future[:, -1].cpu().numpy())


preds = np.vstack(all_preds)
targets = np.vstack(all_targets)

residuals = preds - targets


# ==========================================================
# Overall metrics
# ==========================================================

overall_r2 = r2_score(targets, preds)
overall_rmse = np.sqrt(mean_squared_error(targets, preds))

print("Overall R2:", overall_r2)

with open(os.path.join(OUTPUT_DIR, "metrics", "overall_metrics.txt"), "w") as f:
    f.write(f"Overall R2: {overall_r2}\n")
    f.write(f"Overall RMSE: {overall_rmse}\n")


# ==========================================================
# Per sensor metrics
# ==========================================================

sensor_r2 = []
sensor_rmse = []

for i, name in enumerate(SENSOR_NAMES):
    r2 = r2_score(targets[:, i], preds[:, i])
    rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))

    sensor_r2.append(r2)
    sensor_rmse.append(rmse)


metrics_df = pd.DataFrame(
    {"sensor": SENSOR_NAMES, "r2": sensor_r2, "rmse": sensor_rmse}
)

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "metrics", "per_sensor_metrics.csv"), index=False
)


# ==========================================================
# Sensor ranking plot
# ==========================================================

plt.figure(figsize=(14, 6))

plt.bar(SENSOR_NAMES, sensor_r2)

plt.xticks(rotation=90)

plt.title("Per Sensor R2")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "metrics", "sensor_r2_bar.png"), dpi=200)

plt.close()


# ==========================================================
# Generate plots for ALL sensors
# ==========================================================

print("Generating sensor plots...")

for i, name in enumerate(SENSOR_NAMES):
    # Scatter plot
    plt.figure(figsize=(5, 5))

    plt.scatter(targets[:4000, i], preds[:4000, i], alpha=0.3)

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(name)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "prediction_scatter", f"{name}_scatter.png"), dpi=200
    )

    plt.close()

    # Time series plot
    plt.figure(figsize=(10, 4))

    plt.plot(targets[:500, i], label="True")
    plt.plot(preds[:500, i], label="Pred")

    plt.legend()

    plt.title(name)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "prediction_timeseries", f"{name}_timeseries.png"),
        dpi=200,
    )

    plt.close()

    # Residual histogram
    plt.figure()

    plt.hist(residuals[:, i], bins=80)

    plt.title(name)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "residual_histograms", f"{name}_residual_hist.png"),
        dpi=200,
    )

    plt.close()

    # Residual autocorrelation
    res = residuals[:, i]

    lags = 50

    autocorr = [np.corrcoef(res[:-k], res[k:])[0, 1] for k in range(1, lags)]

    plt.figure()

    plt.plot(autocorr)

    plt.title(f"{name} Residual Autocorr")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "residual_autocorr", f"{name}_autocorr.png"), dpi=200
    )

    plt.close()


# ==========================================================
# Rollout Error Growth
# ==========================================================

print("Evaluating rollout stability")

rollout_errors = []
horizons = []

x_hist, u_hist, u_future, y_future = dataset[0]

x_hist = x_hist.unsqueeze(0).to(DEVICE)
u_hist = u_hist.unsqueeze(0).to(DEVICE)

u_future = u_future[:MAX_ROLLOUT].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    preds_roll = model.rollout(x_hist, u_hist, u_future)

preds_roll = preds_roll.cpu().numpy()[0]
truth_roll = y_future[:MAX_ROLLOUT].numpy()

for h in range(2, MAX_ROLLOUT + 1):
    r2 = r2_score(truth_roll[:h], preds_roll[:h])

    rollout_errors.append(r2)
    horizons.append(h)

plt.figure()

plt.plot(horizons, rollout_errors)

plt.xlabel("Rollout Horizon")
plt.ylabel("R2")

plt.title("Rollout Stability Curve")

plt.grid(True)

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "rollout", "rollout_r2_curve.png"), dpi=200)

plt.close()


# ==========================================================
# Latent space PCA
# ==========================================================

print("Computing latent space PCA")

latents = []

with torch.no_grad():
    for x_hist, u_hist, _, _ in loader:
        x_hist = x_hist.to(DEVICE)
        u_hist = u_hist.to(DEVICE)

        z = model.encoder(x_hist, u_hist)

        latents.append(z.cpu().numpy())


latents = np.vstack(latents)

pca = PCA(n_components=2)

z2 = pca.fit_transform(latents)


plt.figure()

plt.scatter(z2[:, 0], z2[:, 1], alpha=0.2)

plt.title("Latent Space PCA")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "latent_analysis", "latent_pca.png"), dpi=200)

plt.close()


print("\nDiagnostics complete.")
print("Results saved in:", OUTPUT_DIR)
