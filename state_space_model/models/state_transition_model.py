import torch
import torch.nn as nn


class StateTransitionModel(nn.Module):
    def __init__(self, state_dim, control_dim, hidden=256):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, x, u):

        inp = torch.cat([x, u], dim=-1)

        dx = self.net(inp)

        return x + dx
