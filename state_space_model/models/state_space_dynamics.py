import torch
import torch.nn as nn


class StateSpaceDynamics(nn.Module):
    def __init__(self, hidden_dim, control_dim):

        super().__init__()

        self.control_embed = nn.Linear(control_dim, hidden_dim)

        self.A = nn.Linear(hidden_dim, hidden_dim)

        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, u):

        u = self.control_embed(u)

        linear = self.A(h)

        nonlinear = self.residual(torch.cat([h, u], dim=-1))

        return linear + nonlinear
