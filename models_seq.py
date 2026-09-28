"""Sequence models for the Z24 classification task, all working on full-length (B, L=6000, C=27) inputs.

Common interface (so run_prism_z24.py can train any of them):
  forward(x)  -> logits   (x is (B, L, C))
  embed(x)    -> (B, D) features before the classifier head (used for t-SNE)

  prism        PRISM (multi-resolution symmetric CNN, https://github.com/fedezuc/PRISM) - not a sequence model,
               kept as the reference
  ms4n         S4D state-space model following the MS4N description (Algorithm 2 of arXiv:2605.27406):
               linear input projection -> S4D (FFT convolution) -> GELU/dropout -> gated channel mixing (GLU)
               -> LayerNorm -> global average pooling -> MLP. There is no official code, so this is a
               re-implementation from the paper text (S4D-Lin initialisation instead of exact HiPPO-LegS).
  gru / lstm   learned strided-conv stem (NOT resampling; --stem_stride 1 = no reduction) + bidirectional RNN
  cnn_lstm     two conv+pool blocks + LSTM (the 1DCNN-LSTM baseline family)
  transformer  patch embedding over all channels (patch 50, stride 25 -> 239 tokens) + Transformer encoder
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ["prism", "ms4n", "gru", "lstm", "cnn_lstm", "transformer"]


class PrismWrap(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from models.PRISM import Model  # needs the cloned PRISM repo on sys.path

        self.m = Model(cfg)

    def forward(self, x):
        return self.m(x)

    def embed(self, x):
        return self.m.front(x.transpose(1, 2)).mean(dim=1)


# ------------------------------------------------------------------ S4D / MS4N
class S4D(nn.Module):
    """Diagonal state-space layer: y = K * u (FFT convolution) + D*u, then GELU and dropout."""

    def __init__(self, d_model, n_state=64, dropout=0.1, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        n2 = n_state // 2  # complex conjugate pairs
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)                                   # learnable step size
        self.C = nn.Parameter(torch.randn(d_model, n2, 2))                   # complex C stored as (re, im)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, n2)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n2).float().repeat(d_model, 1))  # S4D-Lin
        self.D = nn.Parameter(torch.randn(d_model))
        self.drop = nn.Dropout(dropout)

    def kernel(self, L):
        dt = torch.exp(self.log_dt)                                          # (H,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag                   # (H, N/2) complex
        C = torch.view_as_complex(self.C)
        dtA = A * dt[:, None]
        K = dtA[..., None] * torch.arange(L, device=dt.device)               # (H, N/2, L)
        C = C * (torch.exp(dtA) - 1.0) / A                                   # zero-order-hold discretisation
        return 2 * torch.einsum("hn,hnl->hl", C, torch.exp(K)).real          # (H, L)

    def forward(self, u):                                                    # u: (B, H, L)
        L = u.size(-1)
        k = self.kernel(L)
        n = 2 * L
        y = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :L]
        y = y + u * self.D[None, :, None]
        return self.drop(F.gelu(y))


class MS4N(nn.Module):
    def __init__(self, in_ch, n_cls, d_model=64, n_state=64, layers=1, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)                                # F -> H, decouples from channel count
        self.s4 = nn.ModuleList([S4D(d_model, n_state, dropout) for _ in range(layers)])
        self.mix = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(layers)])   # gated channel mixing
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])            # the "N" in MS4N
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_cls))
        self.residual = layers > 1                                           # paper uses a single block

    def embed(self, x):                                                      # x: (B, L, C)
        h = self.proj(x)
        for s4, mix, norm in zip(self.s4, self.mix, self.norm):
            y = s4(h.transpose(1, 2)).transpose(1, 2)
            a, b = mix(y).chunk(2, dim=-1)
            g = norm(a * torch.sigmoid(b))
            h = h + g if self.residual else g
        return h.mean(dim=1)                                                 # global average pooling over time

    def forward(self, x):
        return self.head(self.embed(x))


# ------------------------------------------------------------------ RNNs
class RNNClassifier(nn.Module):
    def __init__(self, in_ch, n_cls, kind="gru", hidden=64, layers=2, stem_stride=5, dropout=0.1):
        super().__init__()
        s = stem_stride
        self.stem = nn.Sequential(nn.Conv1d(in_ch, hidden, 2 * s + 1, stride=s, padding=s),
                                  nn.BatchNorm1d(hidden), nn.GELU())
        cell = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cell(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                        dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(dropout), nn.Linear(2 * hidden, n_cls))

    def embed(self, x):
        self.rnn.flatten_parameters()
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(h)
        return out.mean(dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


class CNNLSTM(nn.Module):
    def __init__(self, in_ch, n_cls, hidden=64, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 9, padding=4), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(32, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(), nn.MaxPool1d(4))
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(dropout), nn.Linear(2 * hidden, n_cls))

    def embed(self, x):
        self.rnn.flatten_parameters()
        out, _ = self.rnn(self.cnn(x.transpose(1, 2)).transpose(1, 2))
        return out.mean(dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


# ------------------------------------------------------------------ Transformer
class PatchTransformer(nn.Module):
    def __init__(self, in_ch, n_cls, seq_len, patch=50, stride=25, d_model=64, heads=4, layers=3, ff=128,
                 dropout=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        n_tok = (seq_len - patch) // stride + 1
        self.patch_embed = nn.Linear(in_ch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

    def embed(self, x):                                                      # x: (B, L, C)
        p = x.unfold(1, self.patch, self.stride).flatten(2)                  # (B, n_tok, C*patch)
        h = self.enc(self.patch_embed(p) + self.pos)
        return self.norm(h).mean(dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


def build_model(name, cfg, args=None):
    """cfg needs enc_in, num_class, seq_len; args may carry hidden / layers / stem_stride."""
    def g(key, default):  # None / missing on the command line -> the model's own default
        v = getattr(args, key, None) if args is not None else None
        return default if v is None else v

    c, k, L = cfg.enc_in, cfg.num_class, cfg.seq_len
    if name == "prism":
        return PrismWrap(cfg)
    if name == "ms4n":
        return MS4N(c, k, d_model=g("hidden", 64), layers=g("layers", 1))
    if name in ("gru", "lstm"):
        return RNNClassifier(c, k, kind=name, hidden=g("hidden", 64), layers=g("layers", 2),
                             stem_stride=g("stem_stride", 5))
    if name == "cnn_lstm":
        return CNNLSTM(c, k, hidden=g("hidden", 64))
    if name == "transformer":
        return PatchTransformer(c, k, L, d_model=g("hidden", 64), layers=g("layers", 3))
    raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
