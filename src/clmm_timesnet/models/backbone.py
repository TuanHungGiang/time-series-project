import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .timesnet import Model as TimesNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timesnet_config(channels: int, length: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="anomaly_detection", seq_len=length, label_len=0, pred_len=0,
        top_k=3, num_kernels=4, d_model=64, d_ff=64, e_layers=2,
        enc_in=channels, c_out=channels, embed="timeF", freq="h", dropout=0.1,
    )


def reconstruct(model, windows):
    return model(windows.transpose(1, 2), None, None, None).transpose(1, 2)


def fit_timesnet(fit, epochs, batch_size, device):
    channels, length = fit.shape[1:]
    model = TimesNet(timesnet_config(channels, length)).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(fit)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (batch - reconstruct(model, batch)).pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if os.environ.get("CLMM_LOG_EACH_EPOCH") == "1" or epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  epoch={epoch}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    return model.eval().requires_grad_(False)
