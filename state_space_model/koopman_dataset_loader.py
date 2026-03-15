# dataset_loader.py

import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessDataset(Dataset):
    def __init__(self):

        print("Opening dataset...")

        self.X = np.load("X_hist.npy", mmap_mode="r")
        self.U = np.load("U_hist.npy", mmap_mode="r")
        self.Uf = np.load("U_future.npy", mmap_mode="r")
        self.Y = np.load("Y_future.npy", mmap_mode="r")

        self.length = self.X.shape[0]

        print("Samples:", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        x = torch.from_numpy(self.X[idx].copy()).float()
        u = torch.from_numpy(self.U[idx].copy()).float()
        uf = torch.from_numpy(self.Uf[idx].copy()).float()
        y = torch.from_numpy(self.Y[idx].copy()).float()

        return x, u, uf, y
