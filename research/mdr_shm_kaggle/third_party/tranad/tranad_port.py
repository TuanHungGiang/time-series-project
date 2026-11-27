"""Minimal batch-first port of the official TranAD architecture.

Derived from imperial-qore/TranAD (BSD-3-Clause). Evaluation, POT and point
adjustment are deliberately excluded so the shared healthy calibration
protocol controls every baseline.
"""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, length: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        values = torch.zeros(length, 1, dimension)
        values[:, 0, 0::2] = torch.sin(position * div)
        values[:, 0, 1::2] = torch.cos(position * div)
        self.register_buffer("values", values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.values[:x.size(0)])


class TranAD(nn.Module):
    def __init__(self, features: int, window_size: int):
        super().__init__()
        dimension = 2 * features
        self.features = features
        self.position = PositionalEncoding(dimension, 0.1, window_size)
        encoder = nn.TransformerEncoderLayer(dimension, features, 16, 0.1)
        decoder1 = nn.TransformerDecoderLayer(dimension, features, 16, 0.1)
        decoder2 = nn.TransformerDecoderLayer(dimension, features, 16, 0.1)
        self.encoder = nn.TransformerEncoder(encoder, 1)
        self.decoder1 = nn.TransformerDecoder(decoder1, 1)
        self.decoder2 = nn.TransformerDecoder(decoder2, 1)
        self.output = nn.Sequential(nn.Linear(dimension, features), nn.Sigmoid())

    def encode(self, source: torch.Tensor, condition: torch.Tensor, target: torch.Tensor):
        source = self.position(torch.cat([source, condition], dim=2) * math.sqrt(self.features))
        return target.repeat(1, 1, 2), self.encoder(source)

    def forward(self, batch: torch.Tensor):
        # B,T,C -> T,B,C, reconstruct the complete window rather than only
        # its last point, retaining the official two-phase conditioning.
        source = batch.transpose(0, 1)
        target = source
        first = self.output(self.decoder1(*self.encode(source, torch.zeros_like(source), target)))
        condition = (first - source).pow(2)
        second = self.output(self.decoder2(*self.encode(source, condition, target)))
        return first.transpose(0, 1), second.transpose(0, 1)

