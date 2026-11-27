"""Measure the same training objective on held-out healthy windows."""

import numpy as np
import torch

from .models.backbone import reconstruct


def log_epoch(rows, dataset, group, seed, model, epoch, train_loss, val_loss, n_ref, objective):
    if n_ref < 1 or not np.isfinite([train_loss, val_loss]).all():
        raise ValueError(f"Invalid train or validation loss for {dataset}/{group}/{model}/epoch {epoch}")
    rows.append(dict(dataset=dataset, group=group, seed=seed, model=model, epoch=epoch,
                     train_loss=float(train_loss), val_loss=float(val_loss), reference_windows=n_ref,
                     objective=objective))
    step = f"epoch={epoch}" if epoch else "single PCA fit"
    print(f"  {dataset}/{group} {model} {step} "
          f"train_loss={train_loss:.8f} val_loss={val_loss:.8f}", flush=True)


def validation_callback(rows, dataset, group, seed, name, reference, batch_size, device, source):
    def callback(epoch, train_loss, network, edges=None, transform=None):
        values = transform(reference) if transform is not None else reference
        was_training = network.training
        network.eval()
        objective = "ScatterAD objective" if name == "scatterad" else "MSE"
        devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                inputs = torch.from_numpy(values)
                if edges is not None:
                    losses = []
                    for window in inputs:
                        x = window.T.contiguous().to(device)
                        predicted, target = network.model(x, edges)
                        losses.append(network.compute_loss(x, predicted, target, edges))
                    val_loss = torch.stack(losses).mean().item()
                else:
                    scores = []
                    size = 1 if name == "tranad" else batch_size
                    for batch in inputs.to(device).split(size):
                        if name == "timesnet_native":
                            predicted = reconstruct(network, batch)
                        elif name == "cnn_ae_1d":
                            predicted = network(batch)
                        else:
                            predicted = source.model_forward(name, network, batch)
                        scores.append((batch - predicted).square().mean((1, 2)))
                    val_loss = torch.cat(scores).mean().item()
        finally:
            network.train(was_training)
        log_epoch(rows, dataset, group, seed, name, epoch, train_loss, val_loss,
                  len(reference), objective)
    return callback
