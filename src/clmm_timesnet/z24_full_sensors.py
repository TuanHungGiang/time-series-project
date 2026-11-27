"""Keep all structural-response axes in Z24 and remove whole sensor locations."""

import re
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import loadmat

from . import mask_pair_block as windows
from .data.masking import sensor_mask, stable_rng
from .data.z24 import split_training_windows
from .post_block_head import REFERENCE_MASKS
from .whitening_two_bridges_kaggle import Z24WhiteningAudit


RESPONSE_CODE = re.compile(r"(?:[0-9]+|R[123])[LTV]\Z")


def response_labels(path):
    raw = loadmat(path, variable_names=["labelshulp"])["labelshulp"]
    labels = [str(value).strip() for value in raw.ravel()]
    response = [value for value in labels if RESPONSE_CODE.fullmatch(value)]
    if len(response) != len(set(response)) or not response:
        raise ValueError(f"Duplicate or missing response labels in {path}")
    return set(response)


@dataclass
class Layout:
    channels: tuple[str, ...]
    sites: tuple[str, ...]
    site_of_axis: np.ndarray

    @classmethod
    def from_training(cls, label_sets):
        if not label_sets or any(labels != label_sets[0] for labels in label_sets[1:]):
            raise ValueError("Healthy training recordings need a common complete channel roster within each setup")
        channels = tuple(sorted(label_sets[0], key=lambda value: (
            not value.startswith("R"), int(value.removeprefix("R")[:-1]), "VLT".index(value[-1]),
        )))
        required_refs = {"R1V", "R2V", "R2L", "R2T", "R3V"}
        if not required_refs.issubset(channels):
            raise ValueError(f"Missing fixed reference responses from {channels}")
        sites = tuple(dict.fromkeys(name[:-1] for name in channels))
        return cls(channels, sites, np.array([sites.index(c[:-1]) for c in channels]))

    def available_sites(self, axes):
        available = np.zeros(len(self.sites), dtype=bool)
        available[self.site_of_axis[axes]] = True
        return available

    def training_mask(self, rng, rate):
        available = sensor_mask(len(self.sites), rate, rng.permutation(len(self.sites)))
        return available[self.site_of_axis]

    def cases(self, measured, record_id, replicates):
        yield "clean", 0, measured.copy()
        sites = np.flatnonzero(self.available_sites(measured))
        for replicate in replicates:
            order = stable_rng(record_id, "z24-full-physical-site", replicate).permutation(len(sites))
            for rate in windows.RATES:
                surviving = sensor_mask(len(sites), rate, order)
                lost = sites[~surviving]
                yield f"sensor_{rate:.2f}", replicate, measured & ~np.isin(self.site_of_axis, lost)

    def mask_audit(self, measured, kept):
        sites_measured = self.available_sites(measured)
        sites_kept = self.available_sites(kept)
        removed = int((sites_measured & ~sites_kept).sum())
        return dict(
            total_sites=len(self.sites), measured_sites=int(sites_measured.sum()),
            kept_sites=int(sites_kept.sum()), simulated_site_loss=removed / int(sites_measured.sum()),
            simulated_channel_loss=int((measured & ~kept).sum()) / int(measured.sum()),
            natural_missing_channels=" ".join(np.array(self.channels)[~measured]),
            simulated_missing_channels=" ".join(np.array(self.channels)[measured & ~kept]),
            simulated_missing_sites=" ".join(np.array(self.sites)[sites_measured & ~sites_kept]),
        )


class FullSensorWhiteningAudit(Z24WhiteningAudit):
    @torch.no_grad()
    def scores(self, data, kept):
        missing = int((~kept).sum())
        if missing and missing not in self.centers:
            # A fixed number of removed sites may result in several missing-axis
            # counts. Extend the original healthy-reference gain centering to
            # the new count, keeping the fitted gain scale and score weights.
            channels = len(kept)
            order = stable_rng(self.full_group, "post-block-gain-reference", 0).permutation(channels)
            gains = [[] for _ in order]
            num_masks = max(REFERENCE_MASKS, (channels + channels - missing - 1) // (channels - missing))
            for roll in np.arange(num_masks) * channels // num_masks:
                retained = np.ones(channels, dtype=bool)
                retained[np.roll(order, roll)[:missing]] = False
                mask = torch.as_tensor(retained, device=self.detector.device)
                values = np.concatenate([
                    self.batch_signals(chunk, mask)[2] for chunk in self.healthy.split(self.detector.batch_size)
                ])
                for index, column in zip(np.flatnonzero(retained), values.T, strict=True):
                    gains[index].append(column)
            if not all(gains):
                raise ValueError(f"Incomplete healthy reference for {missing} unavailable axes")
            self.centers[missing] = np.array([np.median(np.concatenate(parts)) for parts in gains])
        return super().scores(data, kept)

    def reference_audit(self, reference, group, args):
        self.full_group = group
        super().reference_audit(reference, group, args)


def load_aligned(path, layout, present):
    measured = np.array([channel in present for channel in layout.channels])
    if present.difference(layout.channels) or not measured.any():
        raise ValueError(f"{path.name}: unexpected layout relative to healthy training")
    subset = [c for c, available in zip(layout.channels, measured, strict=True) if available]
    raw = windows.load_z24(path, subset)
    if measured.all():
        return raw, measured
    aligned = np.zeros((len(raw), len(layout.channels), raw.shape[-1]), dtype=raw.dtype)
    aligned[:, measured] = raw
    return aligned, measured


def run_z24_full(args, device, make_block):
    manifest = windows.load_manifest(args.z24_root).assign(sample_id=lambda f: f.record_id)
    rows, masks, fits = [], [], []
    expected_groups = {(i, e) for i in range(1, 10) for e in ("avt", "fvt")}
    grouped = manifest.groupby(["setup", "excitation"], sort=True)
    if set(grouped.groups) != expected_groups:
        raise ValueError("Expected all 18 Z24 setup–excitation groups")
    for index, ((setup, excitation), cohort) in enumerate(grouped):
        group = f"{setup:02d}/{excitation.upper()}"
        train = cohort.loc[cohort.split.eq("train")]
        paths = [Path(path) for path in train.path]
        available = [response_labels(path) for path in paths]
        layout = Layout.from_training(available)
        train_windows = [load_aligned(p, layout, a)[0] for p, a in zip(paths, available, strict=True)]
        fit, reference = split_training_windows(train_windows)
        fit, reference = windows.subset(fit, args.z24_windows), windows.subset(reference, args.z24_windows)
        group_args = copy(args)
        group_args.training_sensor_mask = layout.training_mask
        group_seed = args.seed + 1009 * index
        print(f"Z24 {group}: {len(layout.sites)} sites, {len(layout.channels)} response channels", flush=True)
        block = make_block(fit, reference, group_seed, group_args, device, group)
        fits.append(dict(group=group, seed=group_seed, channels=list(layout.channels),
                         sites=list(layout.sites), fit_windows=len(fit), reference_windows=len(reference),
                         healthy_records=train.record_id.tolist()))
        for idx, row in cohort.loc[cohort.split.isin(["val", "test"])].iterrows():
            path = Path(row.path)
            raw, measured = load_aligned(path, layout, response_labels(path))
            raw = windows.subset(raw, args.z24_windows)
            clean_only = row.split == "val" and row.label == 1
            cases = list(layout.cases(measured, row.record_id, () if clean_only else args.replicates))
            before = len(masks)
            with patch.object(windows, "cases", lambda *_: iter(cases)):
                windows.save_cases(block, raw, cohort.loc[[idx]], row.record_id, group_args, rows, masks, group,
                                   record_level=True, clean_only=clean_only)
            for audit, (_, _, kept) in zip(masks[before:], cases, strict=True):
                audit.update(layout.mask_audit(measured, kept))
    return rows, masks, fits
