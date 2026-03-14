import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self):

        self.X = np.load("X_t.npy", mmap_mode="r")
        self.U = np.load("U_t.npy", mmap_mode="r")
        self.Y = np.load("X_next.npy", mmap_mode="r")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        x = torch.from_numpy(self.X[idx].copy()).float()
        u = torch.from_numpy(self.U[idx].copy()).float()
        y = torch.from_numpy(self.Y[idx].copy()).float()

        return x, u, y
