import numpy as np
import pyreadr
from sklearn.preprocessing import StandardScaler

HORIZON = 20

print("Loading TEP dataset...")

result = pyreadr.read_r("TEP_FaultFree_Training.RData")
key = list(result.keys())[0]

df = result[key]

cols = [c.lower() for c in df.columns]

X_idx = [i for i, c in enumerate(cols) if "xmeas" in c]
U_idx = [i for i, c in enumerate(cols) if "xmv" in c]

X = df.iloc[:, X_idx].values
U = df.iloc[:, U_idx].values

print("States:", X.shape)
print("Controls:", U.shape)

x_scaler = StandardScaler()
u_scaler = StandardScaler()

X = x_scaler.fit_transform(X)
U = u_scaler.fit_transform(U)

X_t = X[:-HORIZON]
U_t = U[:-HORIZON]
X_target = X[HORIZON:]

np.save("X_t.npy", X_t.astype("float32"))
np.save("U_t.npy", U_t.astype("float32"))
np.save("X_target.npy", X_target.astype("float32"))

print("Dataset saved")

print("X_t:", X_t.shape)
print("U_t:", U_t.shape)
print("X_target:", X_target.shape)
