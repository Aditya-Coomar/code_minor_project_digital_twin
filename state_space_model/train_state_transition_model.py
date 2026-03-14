import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from dataset_transition import TransitionDataset
from models.state_transition_model import StateTransitionModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
EPOCHS = 80


dataset = TransitionDataset()

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)

sample = dataset[0]

state_dim = sample[0].shape[0]
control_dim = sample[1].shape[0]

model = StateTransitionModel(state_dim, control_dim).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

criterion = nn.MSELoss()


for epoch in range(EPOCHS):
    preds = []
    targets = []

    for x, u, y in loader:
        x = x.to(DEVICE)
        u = u.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(x, u)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        preds.append(pred.detach().cpu().numpy())
        targets.append(y.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    r2 = r2_score(targets, preds)

    print(f"Epoch {epoch + 1} | R2 {r2:.4f}")
