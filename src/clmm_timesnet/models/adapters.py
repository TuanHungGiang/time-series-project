import torch
from torch import nn


class CrossSensorAdapter(nn.Module):
    def __init__(self, channels: int, length: int, embedding: int, heads: int):
        super().__init__()
        self.channels = channels
        self.length = length
        self.encoder = nn.Sequential(
            nn.Linear(length, 128), nn.GELU(), nn.Linear(128, embedding),
        )
        self.sensor_embedding = nn.Parameter(torch.randn(channels, embedding) * 0.02)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, embedding))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding, nhead=heads, dim_feedforward=embedding * 4,
            dropout=0.10, activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.decoder = nn.Sequential(
            nn.Linear(embedding, 128), nn.GELU(), nn.Linear(128, length),
        )

    def forward(self, values: torch.Tensor, available: torch.Tensor):
        encoded = self.encoder(values) + self.sensor_embedding.unsqueeze(0)
        token = self.missing_token + self.sensor_embedding.unsqueeze(0)
        encoded = available.unsqueeze(-1) * encoded + (1.0 - available.unsqueeze(-1)) * token
        return self.decoder(self.context(encoded))


class ObservedAttentionAdapter(CrossSensorAdapter):
    """Keep missing sensor queries, but exclude them from attention keys and values."""

    def forward(self, values, available):
        visible = available.bool()
        positions = self.sensor_embedding.unsqueeze(0)
        tokens = torch.where(visible.unsqueeze(-1), self.encoder(values), self.missing_token) + positions
        return self.decoder(self.context(tokens, src_key_padding_mask=~visible))
