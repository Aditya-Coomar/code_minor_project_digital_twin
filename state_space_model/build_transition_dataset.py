import numpy as np
import pyreadr
from sklearn.preprocessing import StandardScaler

HORIZON = 20

result = pyreadr.read_r("TEP_FaultFree_Training.RData")
key = list(result.keys())[0]
df = result[key]

cols = [c.lower() for c in df.columns]

X_idx = [i for i, c in enumerate(cols) if "xmeas" in c]
U_idx = [i for i, c in enumerate(cols) if "xmv" in c]

X = df.iloc[:, X_idx].values
U = df.iloc[:, U_idx].values

x_scaler = StandardScaler()
u_scaler = StandardScaler()

X = x_scaler.fit_transform(X)
U = u_scaler.fit_transform(U)

X_t = X[:-1]
U_t = U[:-1]
X_next = X[1:]

np.save("X_t.npy", X_t.astype("float32"))
np.save("U_t.npy", U_t.astype("float32"))
np.save("X_next.npy", X_next.astype("float32"))

print("Dataset ready")
