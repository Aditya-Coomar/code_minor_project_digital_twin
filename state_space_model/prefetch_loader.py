import torch


class PrefetchLoader:

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):

        for X, U, y in self.loader:

            yield (
                X.to(self.device, non_blocking=True),
                U.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True),
            )

    def __len__(self):
        return len(self.loader)