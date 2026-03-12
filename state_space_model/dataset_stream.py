import numpy as np
import torch
from torch.utils.data import Dataset


class TEPDataset(Dataset):
    def __init__(self, X_path, U_path, y_path):

        self.X = np.load(X_path, mmap_mode="r")
        self.U = np.load(U_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        self.length = self.X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        X = torch.from_numpy(self.X[idx].copy()).float()
        U = torch.from_numpy(self.U[idx].copy()).float()
        y = torch.from_numpy(self.y[idx].copy()).float()

        return X, U, y
