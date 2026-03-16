# models/koopman_twin.py
# Koopman neural digital twin with residual (delta) decoder

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """Bidirectional GRU encoder for process history."""

    def __init__(self, state_dim, control_dim, latent):

        super().__init__()

        self.input_proj = nn.Linear(state_dim + control_dim, latent)

        self.gru = nn.GRU(
            latent, latent // 2, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, latent),
        )

    def forward(self, x, u):

        inp = torch.cat([x, u], dim=-1)
        inp = self.input_proj(inp)

        out, _ = self.gru(inp)

        z = out[:, -1]

        return self.out_proj(z)


class KoopmanDynamics(nn.Module):
    """Linear Koopman operator + scaled nonlinear residual."""

    RESIDUAL_SCALE = 0.3

    def __init__(self, latent, control_dim):

        super().__init__()

        self.A = nn.Linear(latent, latent, bias=False)
        self.B = nn.Linear(control_dim, latent, bias=False)

        self.linear_norm = nn.LayerNorm(latent)

        self.residual = nn.Sequential(
            nn.Linear(latent + control_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent),
        )

        self.residual_norm = nn.LayerNorm(latent)

        nn.init.orthogonal_(self.A.weight)

    def forward(self, z, u):

        linear = self.linear_norm(self.A(z) + self.B(u))

        res = self.residual_norm(self.residual(torch.cat([z, u], dim=-1)))

        return linear + self.RESIDUAL_SCALE * res


class ResidualDecoder(nn.Module):
    """Predicts state CHANGE (delta) from latent + reference state.

    x_pred = x_ref + delta_net(z, x_ref)

    Last layer initialized to zero so initial output = x_ref (persistence baseline).
    """

    def __init__(self, latent, state_dim):

        super().__init__()

        self.norm = nn.LayerNorm(latent)

        self.delta_net = nn.Sequential(
            nn.Linear(latent + state_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, state_dim),
        )

        # Zero-init last layer → model starts at persistence baseline
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

    def forward(self, z, x_ref):

        z = self.norm(z)
        delta = self.delta_net(torch.cat([z, x_ref], dim=-1))
        return x_ref + delta


class KoopmanTwin(nn.Module):
    def __init__(self, state_dim, control_dim, latent=256):

        super().__init__()

        self.encoder = HistoryEncoder(state_dim, control_dim, latent)
        self.dynamics = KoopmanDynamics(latent, control_dim)
        self.decoder = ResidualDecoder(latent, state_dim)

    def rollout(self, x_hist, u_hist, u_future):

        z = self.encoder(x_hist, u_hist)

        x_prev = x_hist[:, -1]

        preds = []

        for t in range(u_future.shape[1]):
            u = u_future[:, t]

            z = self.dynamics(z, u)

            x_pred = self.decoder(z, x_prev)

            preds.append(x_pred)

            x_prev = x_pred

        preds = torch.stack(preds, dim=1)

        return preds
