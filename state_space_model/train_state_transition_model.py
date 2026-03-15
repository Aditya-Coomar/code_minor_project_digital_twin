import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from dataset_transition import TransitionDataset
from models.state_transition_model import StateTransitionModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


BATCH_SIZE = 64
EPOCHS = 80


dataset = TransitionDataset()

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)


sample = dataset[0]

state_dim = sample[0].shape[0]
control_dim = sample[1].shape[0]

print("State dim:", state_dim)
print("Control dim:", control_dim)


model = StateTransitionModel(state_dim, control_dim).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

criterion = nn.MSELoss()


def rollout(x, u, horizon):

    preds = []

    for _ in range(horizon):
        x = model(x, u)

        preds.append(x)

    return torch.stack(preds, dim=1)


def rollout_schedule(epoch):

    if epoch < 10:
        return 1
    elif epoch < 30:
        return 5
    elif epoch < 60:
        return 10
    else:
        return 20


for epoch in range(EPOCHS):
    horizon = rollout_schedule(epoch)

    preds_all = []
    targets_all = []

    for x, u, y in loader:
        x = x.to(DEVICE)
        u = u.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = rollout(x, u, horizon)

        target = y.unsqueeze(1).repeat(1, horizon, 1)

        loss = criterion(pred, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds_all.append(pred[:, -1].detach().cpu().numpy())
        targets_all.append(y.cpu().numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    r2 = r2_score(targets_all, preds_all)

    print(f"Epoch {epoch + 1} | Horizon {horizon} | R2 {r2:.4f}")
