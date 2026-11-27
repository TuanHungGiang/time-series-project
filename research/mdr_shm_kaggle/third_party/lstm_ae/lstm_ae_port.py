"""Batch-safe port of the LSTM-AE in the supplied TSAD repository."""
import torch
from torch import nn


class LSTMAE(nn.Module):
    def __init__(self, features: int, window_size: int, hidden: int = 64):
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.LSTM(features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x)
        latent = hidden[-1].unsqueeze(1).repeat(1, self.window_size, 1)
        decoded, _ = self.decoder(latent, (hidden, cell))
        return self.output(decoded)

