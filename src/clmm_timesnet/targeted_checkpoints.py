"""Portable, inference-ready checkpoint files for the targeted baseline sweep."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .models import cnn_ae_1d


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2), encoding="utf-8")
    temporary.replace(path)


def architecture(source, name, channels, points):
    if name == cnn_ae_1d.MODEL_ID:
        return cnn_ae_1d.model_config(channels, points)
    if name == "timesnet":
        return vars(source.timesnet_config(channels, points))
    if name == "moderntcn":
        return vars(source.moderntcn_config(channels, points))
    if name == "usad":
        width = channels * points
        return {"window_size": width, "latent_size": max(5, width // 8), "loss_weights": [0.5, 0.5]}
    if name == "scatterad":
        return {"input_dim": channels, "hidden_dim": 64, "num_layers": 2, "edge_radius": 2}
    if name == "pca_spe":
        return {"variance_retained": 0.95, "svd_solver": "full"}
    return {"channels": channels, "points": points}


def make_network(source, name, channels, points, config):
    from types import SimpleNamespace
    if name == cnn_ae_1d.MODEL_ID:
        return cnn_ae_1d.ConvAutoencoder1D(channels)
    if name == "usad":
        return source.UsadModel(config["window_size"], config["latent_size"])
    if name == "lstm_ae":
        return source.LSTMAE(channels, points)
    if name == "tranad":
        return source.TranAD(channels, points)
    if name == "timesnet":
        return source.TimesNet(SimpleNamespace(**config))
    if name == "moderntcn":
        return source.ModernTCN(SimpleNamespace(**config))
    if name == "scatterad":
        from third_party.scatterad import Model
        return Model(input_dim=channels, hidden_dim=config["hidden_dim"], num_layers=config["num_layers"])
    raise KeyError(name)


class Scorer:
    def __init__(self, source, name, state, batch_size, device, scaler=None, edges=None):
        self.source = source
        self.name = name
        self.state = state
        self.batch = batch_size
        self.device = device
        self.scaler = scaler
        self.edges = edges

    def __call__(self, values):
        if self.name == "pca_spe":
            flat = values.reshape(len(values), -1)
            reconstructed = self.state.inverse_transform(self.state.transform(flat))
            return np.mean((flat - reconstructed) ** 2, axis=1)
        data = values
        if self.scaler is not None:
            low, high = self.scaler
            data = np.clip((values - low) / np.maximum(high - low, 1e-6), 0, 1).astype(np.float32)
        if self.name == "usad":
            return self.source.score_usad(self.state, data, self.batch, self.device)
        if self.name == "scatterad":
            return self.score_scatter(data)
        if self.name == cnn_ae_1d.MODEL_ID:
            return cnn_ae_1d.score(self.state, data, self.batch, self.device)
        return self.source.score_reconstructor(self.name, self.state, data, self.batch, self.device)

    @torch.no_grad()
    def score_scatter(self, samples):
        pieces = []
        for batch in torch.from_numpy(samples).split(self.batch):
            size, channels, points = batch.shape
            x = batch.to(self.device).transpose(1, 2).reshape(size * points, channels)
            edges = self.edges[None] + (torch.arange(size, device=self.device) * points)[:, None, None]
            edges = edges.transpose(0, 1).reshape(2, -1)
            hidden, _ = self.state.model.online_encoder(x, edges)
            hidden = hidden.reshape(size, points, -1)
            temporal = (hidden[:, 1:] - hidden[:, :-1]).square().mean(-1)
            temporal = torch.nn.functional.pad(temporal, (0, 1))
            centers = self.state.model.center.unsqueeze(0).expand(size, -1, -1)
            distance = torch.cdist(hidden, centers).min(dim=-1).values
            pieces.append((distance + temporal).mean(1).cpu().numpy())
        return np.concatenate(pieces)


def fit_scorer(source, name, values, epochs, batch_size, device):
    scale = source.FitMinMax(values) if name in ("usad", "tranad") else None
    fit = scale.transform(values) if scale is not None else values
    scaler = (scale.low, scale.high) if scale is not None else None
    edges = None
    if name == "pca_spe":
        state = PCA(n_components=0.95, svd_solver="full").fit(values.reshape(len(values), -1))
    elif name == "usad":
        state = source.fit_usad(fit, epochs, batch_size, device)
    elif name == "scatterad":
        state, edges = source.fit_scatterad(fit, epochs, device)
    elif name == cnn_ae_1d.MODEL_ID:
        state = cnn_ae_1d.fit(fit, epochs, batch_size, device)
    else:
        state = source.fit_reconstructor(name, fit, epochs, batch_size, device)
    scorer = Scorer(source, name, state, batch_size, device, scaler, edges)
    if name == "scatterad":
        inputs = values[:min(3, len(values))]
        np.testing.assert_allclose(scorer(inputs), source.score_scatterad((state, edges), inputs, device),
                                   rtol=5e-5, atol=5e-6)
    return scorer


def write_checkpoint(path, scorer, group, dataset, seed, effective_seed, epochs, reference, run_protocol):
    training_input = group.train
    channels, points = training_input.shape[1:]
    config = architecture(scorer.source, scorer.name, channels, points)
    if scorer.name == "pca_spe":
        fields = ("components_", "explained_variance_", "mean_")
        state = {key: torch.from_numpy(getattr(scorer.state, key)) for key in fields}
        shapes = {key: list(value.shape) for key, value in state.items()}
        layers = [{"name": key, "shape": shape} for key, shape in shapes.items()]
        parameters = sum(value.numel() for value in state.values())
    else:
        state = scorer.state.state_dict()
        parameters = sum(parameter.numel() for parameter in scorer.state.parameters())
        layers = [
            {"name": name, "type": type(layer).__name__,
             "parameters": {k: list(v.shape) for k, v in layer.named_parameters(recurse=False)}}
            for name, layer in scorer.state.named_modules()
        ]
    estimated = sum(value.numel() * value.element_size() for value in state.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < estimated + 128 * 1024**2:
        raise OSError(f"Insufficient space for {path.name}. Needs about {estimated / 1024**3:.2f} GiB plus reserve.")
    probe = training_input[:min(scorer.batch, len(training_input), 3)]
    metadata = {
        "format": 1, "model": scorer.name, "dataset": dataset, "group": group.name,
        "seed": seed, "effective_seed": effective_seed, "epochs": epochs, "batch_size": scorer.batch,
        "sensors": list(group.sensors), "points": points, "model_config": config,
        "parameter_count": parameters, "layers": layers, "healthy_score_reference": reference,
        "protocol": run_protocol, "torch_version": str(torch.__version__),
        "input_hz": 50.0 if dataset == "Z24" else 12.5,
    }
    package = {
        "metadata": metadata, "weights": {key: value.detach().cpu() for key, value in state.items()},
        "normalizer_mean": torch.from_numpy(group.normalizer.mean),
        "normalizer_scale": torch.from_numpy(group.normalizer.scale),
        "secondary_scale": None if scorer.scaler is None else [torch.from_numpy(x) for x in scorer.scaler],
        "edges": None if scorer.edges is None else scorer.edges.cpu(),
        "probe": torch.from_numpy(probe.copy()), "probe_scores": torch.from_numpy(scorer(probe)),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(package, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), metadata)


def load_checkpoint(path, source, device):
    package = torch.load(path, map_location="cpu", weights_only=True)
    meta = package["metadata"]
    if meta["format"] != 1:
        raise ValueError(f"Unsupported checkpoint format {meta['format']}")
    name, channels, points = meta["model"], len(meta["sensors"]), meta["points"]
    if name == "pca_spe":
        net = PCA(n_components=0.95, svd_solver="full")
        net.n_features_in_ = channels * points
        for key, value in package["weights"].items():
            setattr(net, key, value.numpy())
    else:
        net = make_network(source, name, channels, points, meta["model_config"])
        net.load_state_dict(package["weights"], strict=True)
        net.to(device).eval().requires_grad_(False)
    secondary = package["secondary_scale"]
    scaler = None if secondary is None else tuple(x.numpy() for x in secondary)
    edges = package["edges"]
    scorer = Scorer(source, name, net, meta["batch_size"], device, scaler,
                    None if edges is None else edges.to(device))
    return package, scorer


def verify_checkpoint(package, scorer):
    actual = scorer(package["probe"].numpy())
    np.testing.assert_allclose(actual, package["probe_scores"].numpy(), rtol=5e-4, atol=5e-6)


def main():
    parser = argparse.ArgumentParser(description="Score preprocessed windows with a saved baseline checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True, help="NPY [N,C,T] after detrending/resampling, before z-score")
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    import run_hellbridge_source_baselines as source
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    package, scorer = load_checkpoint(args.checkpoint, source, device)
    verify_checkpoint(package, scorer)
    meta = package["metadata"]
    x = np.load(args.windows, allow_pickle=False).astype(np.float32)
    sensors = meta["sensors"]
    if x.ndim != 3 or not len(x) or x.shape[1:] != (len(sensors), meta["points"]):
        raise ValueError(f"Expected [N,{len(sensors)},{meta['points']}], got {x.shape}")
    requested = set(args.drop)
    if requested.difference(sensors) or len(requested) == len(sensors):
        raise ValueError("Use existing sensor labels and leave at least one channel")
    values = ((x - package["normalizer_mean"].numpy()) / package["normalizer_scale"].numpy()).astype(np.float32)
    values[:, [sensors.index(label) for label in requested], :] = 0.0
    residual = scorer(values)
    scores = residual.astype(np.float64)
    if meta["healthy_score_reference"] is not None:
        center, scale = meta["healthy_score_reference"]
        scores = (np.log1p(np.maximum(scores, 0)) - center) / scale
    calibration = args.checkpoint.parent / "calibration.json"
    rule = json.loads(calibration.read_text(encoding="utf-8")) if calibration.is_file() else {}
    tail = rule.get("score_tail", "upper")
    if tail == "lower":
        scores = -scores
    elif tail == "two_sided":
        scores = np.abs(scores)
    output = pd.DataFrame({"window": np.arange(len(scores)), "raw_score": residual,
                           "anomaly_score": scores, "score_tail": tail})
    if rule.get("score_unit") == "window":
        output["threshold"] = rule["threshold"]
        output["predicted_damage"] = scores >= rule["threshold"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"CHECKPOINT_RESCORE_PASS output={args.output}", flush=True)


if __name__ == "__main__":
    main()
