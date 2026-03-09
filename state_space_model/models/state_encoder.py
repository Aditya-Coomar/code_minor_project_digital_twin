import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=256):

        super().__init__()

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.control_proj = nn.Linear(control_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
        )

        self.out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, X_seq, U_seq):

        X = self.state_proj(X_seq)
        U = self.control_proj(U_seq)

        x = torch.cat([X, U], dim=-1)

        _, (h, _) = self.lstm(x)

        h = torch.cat([h[-2], h[-1]], dim=-1)

        return self.out(h)
