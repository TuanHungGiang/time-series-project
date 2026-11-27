"""Multichannel PyTorch adaptation of the Keras time-series CNN autoencoder."""

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

MODEL_ID = "cnn_ae_1d"
REFERENCE = "https://keras.io/examples/timeseries/timeseries_anomaly_detection/"
REFERENCE_CODE = "https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_anomaly_detection.py"


def model_config(channels, points):
    return {
        "channels": channels, "points": points, "encoder_widths": [32, 16],
        "decoder_widths": [16, 32], "kernel_size": 7, "stride": 2, "dropout": 0.2,
        "training_loss": "MSE", "window_score": "MAE over channels and time",
        "optimizer": "Adam", "learning_rate": 0.001, "reference": REFERENCE, "reference_code": REFERENCE_CODE,
    }


class ConvAutoencoder1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ConstantPad1d((2, 3), 0),
            nn.Conv1d(channels, 32, 7, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConstantPad1d((2, 3), 0),
            nn.Conv1d(32, 16, 7, stride=2),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose1d(16, 16, 7, stride=2, padding=2)
        self.dropout = nn.Dropout(0.2)
        self.up2 = nn.ConvTranspose1d(16, 32, 7, stride=2, padding=2)
        self.output = nn.ConvTranspose1d(32, channels, 7, padding=3)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, values):
        latent = self.encoder(values)
        # Match SAME geometry at even input lengths by trimming the extra sample on the right.
        decoded = self.dropout(self.up1(latent)[..., :-1].relu())
        decoded = self.up2(decoded)[..., :-1].relu()
        return self.output(decoded)


def fit(values, epochs, batch_size, device, epoch_callback=None):
    if values.shape[-1] % 4:
        raise ValueError("CNN-AE window lengths must be divisible by 4")
    model = ConvAutoencoder1D(values.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
    batches = DataLoader(torch.from_numpy(values), batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        if epoch_callback is not None:
            epoch_callback(epoch, torch.stack(losses).mean().item(), model)
        elif os.environ.get("CLMM_LOG_EACH_EPOCH") == "1" or epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch={epoch}/{epochs} loss={torch.stack(losses).mean().item():.6f}", flush=True)
    return model.eval()


@torch.no_grad()
def score(model, values, batch_size, device):
    scores = []
    for batch in torch.from_numpy(values).split(batch_size):
        batch = batch.to(device)
        scores.append((batch - model(batch)).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)
