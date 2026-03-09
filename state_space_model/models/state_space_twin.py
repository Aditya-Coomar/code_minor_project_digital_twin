import torch
import torch.nn as nn

from models.state_encoder import StateEncoder
from models.state_space_dynamics import StateSpaceDynamics
from models.prediction_head import PredictionHead


class StateSpaceTwin(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=384):

        super().__init__()

        self.encoder = StateEncoder(state_dim, control_dim, hidden_dim)

        self.dynamics = StateSpaceDynamics(hidden_dim, control_dim)

        self.head = PredictionHead(hidden_dim, state_dim)

    def forward(self, X_seq, U_seq):

        h = self.encoder(X_seq, U_seq)

        u_last = U_seq[:, -1, :]

        h_next = self.dynamics(h, u_last)

        x_next = self.head(h_next)

        return x_next
