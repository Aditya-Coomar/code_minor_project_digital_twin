# build_koopman_dataset.py
# Builds sliding-window dataset from TEP fault-free training data
# RESPECTS simulation run boundaries — no cross-run contamination

import numpy as np
import pyreadr
from sklearn.preprocessing import StandardScaler

HISTORY = 30
HORIZON = 20

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
        cl = c.lower()
        if cl in ("run", "run_id", "sim", "sim_id", "simulation"):
            run_col = c
            break

# ==========================================================
# Extract state and control columns
# ==========================================================

X_idx = [i for i, c in enumerate(cols) if "xmeas" in c]
U_idx = [i for i, c in enumerate(cols) if "xmv" in c]

X = df.iloc[:, X_idx].values
U = df.iloc[:, U_idx].values

print("States:", X.shape)
print("Controls:", U.shape)

# Global standardization
x_scaler = StandardScaler()
u_scaler = StandardScaler()

X = x_scaler.fit_transform(X)
U = u_scaler.fit_transform(U)

# ==========================================================
# Determine simulation runs
# ==========================================================

if run_col is not None:
    run_ids = df[run_col].values
    unique_runs = np.unique(run_ids)
    print(f"Found {len(unique_runs)} simulation runs (column: '{run_col}')")
else:
    # Infer: TEP standard is 500 steps per run
    n_total = len(X)
    steps_per_run = 500
    n_runs = n_total // steps_per_run
    run_ids = np.repeat(np.arange(n_runs), steps_per_run)
    unique_runs = np.arange(n_runs)
    print(f"No run column found. Inferred {n_runs} runs of {steps_per_run} steps each.")

# ==========================================================
# Build windows WITHIN each simulation run
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

print(f"\nDataset built: {X_hist.shape[0]} clean samples")
print("History:", X_hist.shape)
print("Future controls:", U_future.shape)
print("Future states:", Y_future.shape)
