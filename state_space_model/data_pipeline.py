import numpy as np
import pandas as pd
import torch
import pyreadr

from sklearn.preprocessing import StandardScaler


class TEPDataProcessor:

    def __init__(self, seq_len=200):

        self.seq_len = seq_len
        self.x_scaler = StandardScaler()
        self.u_scaler = StandardScaler()

    def load_rdata(self, path):

        result = pyreadr.read_r(path)

        # Extract dataframe regardless of object name
        df = list(result.values())[0]

        df = pd.DataFrame(df)

        return df

    def split_variables(self, df):

        state_cols = [c for c in df.columns if "xmeas_" in c]
        control_cols = [c for c in df.columns if "xmv_" in c]

        X = df[state_cols].values
        U = df[control_cols].values

        return X, U

    def normalize(self, X, U):

        X = self.x_scaler.fit_transform(X)
        U = self.u_scaler.fit_transform(U)

        return X, U

    def create_sequences(self, df):

        state_cols = [c for c in df.columns if "xmeas" in c.lower()]
        control_cols = [c for c in df.columns if "xmv" in c.lower()]

        X_seq = []
        U_seq = []
        y = []

        for run in df["simulationRun"].unique():

            run_df = df[df["simulationRun"] == run]

            X = run_df[state_cols].values
            U = run_df[control_cols].values

            X = self.x_scaler.fit_transform(X)
            U = self.u_scaler.fit_transform(U)

            for i in range(self.seq_len, len(X) - 1):

                X_seq.append(X[i-self.seq_len:i])
                U_seq.append(U[i-self.seq_len:i])
                y.append(X[i+1])

        return (
            np.array(X_seq),
            np.array(U_seq),
            np.array(y)
        )

    def process(self, path):

        df = self.load_rdata(path)

        return self.create_sequences(df)