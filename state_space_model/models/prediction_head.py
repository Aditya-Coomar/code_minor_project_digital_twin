import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim, state_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, h):

        return self.net(h)
