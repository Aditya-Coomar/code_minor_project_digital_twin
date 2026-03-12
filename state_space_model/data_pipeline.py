import numpy as np
import pyreadr
from sklearn.preprocessing import StandardScaler


class TEPDataProcessor:
    def __init__(self, seq_len=200):
        self.seq_len = seq_len
        self.x_scaler = StandardScaler()
        self.u_scaler = StandardScaler()

    def load_rdata(self, file_path):

        result = pyreadr.read_r(file_path)
        df = result[None]

        X = df.iloc[:, :41].values.astype(np.float32)
        U = df.iloc[:, 41:].values.astype(np.float32)

        return X, U

    def normalize(self, X, U):

        X = self.x_scaler.fit_transform(X)
        U = self.u_scaler.fit_transform(U)

        return X, U

    def process(self, file_path, horizon):

        X, U = self.load_rdata(file_path)
        X, U = self.normalize(X, U)

        X_seq = []
        U_seq = []
        y = []

        for i in range(self.seq_len, len(X) - horizon):
            X_seq.append(X[i - self.seq_len : i])
            U_seq.append(U[i - self.seq_len : i])
            y.append(X[i + horizon])

        return (
            np.array(X_seq, dtype=np.float32),
            np.array(U_seq, dtype=np.float32),
            np.array(y, dtype=np.float32),
        )
