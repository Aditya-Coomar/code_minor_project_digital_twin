import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim):

        super().__init__()

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.control_proj = nn.Linear(control_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, X_seq, U_seq):

        X = self.state_proj(X_seq)
        U = self.control_proj(U_seq)

        inp = torch.cat([X, U], dim=-1)

        _, (h, _) = self.lstm(inp)

        h = torch.cat([h[-2], h[-1]], dim=-1)

        return self.out(h)


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


class StateSpaceTwin(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=384):

        super().__init__()

        self.encoder = StateEncoder(state_dim, control_dim, hidden_dim)

        self.dynamics = StateSpaceDynamics(hidden_dim, control_dim)

        self.head = nn.Linear(hidden_dim, state_dim)

    def forward(self, X_seq, U_seq, pred_horizon):

        h = self.encoder(X_seq, U_seq)

        x = X_seq[:, -1, :]

        for _ in range(pred_horizon):
            u = U_seq[:, -1, :]

            h = self.dynamics(h, u)

            dx = self.head(h)

            x = x + dx

        return x
