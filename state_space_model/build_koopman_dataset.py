# build_koopman_dataset.py
# Builds sliding-window dataset from TEP fault-free training data
# - Respects simulation run boundaries
# - Smooths state measurements to remove sensor noise

import numpy as np
import pyreadr
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d

HISTORY = 30
HORIZON = 20
SMOOTH_WINDOW = 5

print("Loading TEP dataset...")

result = pyreadr.read_r("TEP_FaultFree_Training.RData")
key = list(result.keys())[0]
df = result[key]

cols = [c.lower() for c in df.columns]

# ==========================================================
# Detect simulation run column
# ==========================================================

run_col = None
for c in df.columns:
    cl = c.lower().replace("_", "").replace(" ", "")
    if "simulationrun" in cl or "simrun" in cl:
        run_col = c
        break

if run_col is None:
    for c in df.columns:
        if c.lower() in ("run", "run_id", "sim", "sim_id"):
            run_col = c
            break

# ==========================================================
# Extract state and control columns
# ==========================================================

X_idx = [i for i, c in enumerate(cols) if "xmeas" in c]
U_idx = [i for i, c in enumerate(cols) if "xmv" in c]

X_raw = df.iloc[:, X_idx].values.astype(np.float64)
U_raw = df.iloc[:, U_idx].values.astype(np.float64)

print("States:", X_raw.shape)
print("Controls:", U_raw.shape)

# ==========================================================
# Determine simulation runs
# ==========================================================

if run_col is not None:
    run_ids = df[run_col].values
    unique_runs = np.unique(run_ids)
    print(f"Found {len(unique_runs)} simulation runs (column: '{run_col}')")
else:
    n_total = len(X_raw)
    steps_per_run = 500
    n_runs = n_total // steps_per_run
    run_ids = np.repeat(np.arange(n_runs), steps_per_run)
    unique_runs = np.arange(n_runs)
    print(f"Inferred {n_runs} runs of {steps_per_run} steps each")

# ==========================================================
# Smooth state measurements within each run (removes sensor noise)
# ==========================================================

print(f"Smoothing states with window={SMOOTH_WINDOW}...")

X_smooth = np.empty_like(X_raw)
for run in unique_runs:
    mask = run_ids == run
    X_smooth[mask] = uniform_filter1d(
        X_raw[mask], size=SMOOTH_WINDOW, axis=0, mode="nearest"
    )

# ==========================================================
# Standardize AFTER smoothing
# ==========================================================

x_scaler = StandardScaler()
u_scaler = StandardScaler()

X = x_scaler.fit_transform(X_smooth).astype(np.float32)
U = u_scaler.fit_transform(U_raw).astype(np.float32)

# ==========================================================
# Build windows within each simulation run
# ==========================================================

X_hist = []
U_hist = []
U_future = []
Y_future = []

for run in unique_runs:
    mask = run_ids == run
    run_X = X[mask]
    run_U = U[mask]

    n = len(run_X)
    if n < HISTORY + HORIZON:
        continue

    for i in range(HISTORY, n - HORIZON):
        X_hist.append(run_X[i - HISTORY : i])
        U_hist.append(run_U[i - HISTORY : i])
        U_future.append(run_U[i : i + HORIZON])
        Y_future.append(run_X[i : i + HORIZON])

X_hist = np.array(X_hist, dtype="float32")
U_hist = np.array(U_hist, dtype="float32")
U_future = np.array(U_future, dtype="float32")
Y_future = np.array(Y_future, dtype="float32")

np.save("X_hist.npy", X_hist)
np.save("U_hist.npy", U_hist)
np.save("U_future.npy", U_future)
np.save("Y_future.npy", Y_future)

print(f"\nDataset built: {X_hist.shape[0]} clean, smoothed samples")
print("History:", X_hist.shape)
print("Future controls:", U_future.shape)
print("Future states:", Y_future.shape)
