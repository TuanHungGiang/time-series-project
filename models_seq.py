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
  mamba        Selective state-space model (Mamba / S6, Gu & Dao 2023-2024; the architecture behind most
               2024-2026 sequence SOTA - unlike S4/S4D its A, B, C, dt depend on the input at every
               timestep). Implemented as a genuine step-by-step recurrence (matching the paper's equations)
               rather than the authors' hardware-aware CUDA scan, which needs a Linux/CUDA build environment
               this project does not assume. Correctness was checked by overfitting 32 samples (100% in ~10
               epochs). A learned strided-conv stem (--stem_stride, default 25) shortens the sequence the
               scan runs over; a Python-level sequential loop's runtime grows worse than linearly with its
               length (measured on an RTX 2050: ~0.2s/batch at length 120, ~8.4s/batch at length 1200), so
               keep --stem_stride at 20+ unless you have time to spare.
  gru / lstm   learned strided-conv stem (NOT resampling; --stem_stride 1 = no reduction) + bidirectional RNN
  cnn_lstm     two conv+pool blocks + LSTM (the 1DCNN-LSTM baseline family)
  transformer  patch embedding over all channels (patch 50, stride 25 -> 239 tokens) + Transformer encoder
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ["prism", "ms4n", "mamba", "gru", "lstm", "cnn_lstm", "transformer"]


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


# ------------------------------------------------------------------ Mamba (selective SSM / S6)
class MambaBlock(nn.Module):
    """One Mamba layer: input-dependent (A, B, C, dt) instead of S4D's fixed A, C.

    Pre-norm residual block. Runs the recurrence h_t = deltaA_t * h_t-1 + deltaB_t * x_t with a plain
    Python loop over time, which is exact but O(L) sequential steps - fine once the stem has shortened L.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        d_inner = expand * d_model
        dt_rank = max(d_model // 16, 1)
        self.d_inner, self.d_state = d_inner, d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        # S4D-real initialisation for A: one decay rate per state slot, shared across channels at init
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, u):  # u: (B, L, d_model)
        B_, L, _ = u.shape
        x, z = self.in_proj(self.norm(u)).chunk(2, dim=-1)          # each (B, L, d_inner)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2))  # causal depthwise conv
        dt_raw, Bs, Cs = self.x_proj(x).split([self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))                       # (B, L, d_inner), input-dependent step size
        A = -torch.exp(self.A_log)                                  # (d_inner, N), always negative -> stable
        deltaA = torch.exp(dt.unsqueeze(-1) * A)                    # (B, L, d_inner, N)
        deltaBx = (dt * x).unsqueeze(-1) * Bs.unsqueeze(2)          # (B, L, d_inner, N)
        h = x.new_zeros(B_, self.d_inner, self.d_state)
        ys = []
        for t in range(L):                                          # sequential selective scan
            h = deltaA[:, t] * h + deltaBx[:, t]
            ys.append(torch.einsum("bhn,bn->bh", h, Cs[:, t]))
        y = torch.stack(ys, dim=1) + x * self.D                     # (B, L, d_inner)
        return u + self.drop(self.out_proj(y * F.silu(z)))


class MambaClassifier(nn.Module):
    def __init__(self, in_ch, n_cls, hidden=64, layers=2, d_state=16, stem_stride=5, dropout=0.1):
        super().__init__()
        s = stem_stride
        self.stem = nn.Sequential(nn.Conv1d(in_ch, hidden, 2 * s + 1, stride=s, padding=s),
                                  nn.BatchNorm1d(hidden), nn.GELU())
        self.blocks = nn.ModuleList([MambaBlock(hidden, d_state=d_state, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_cls))

    def embed(self, x):
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h).mean(dim=1)

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
    if name == "mamba":
        return MambaClassifier(c, k, hidden=g("hidden", 64), layers=g("layers", 2), stem_stride=g("stem_stride", 25))
    if name in ("gru", "lstm"):
        return RNNClassifier(c, k, kind=name, hidden=g("hidden", 64), layers=g("layers", 2),
                             stem_stride=g("stem_stride", 5))
    if name == "cnn_lstm":
        return CNNLSTM(c, k, hidden=g("hidden", 64))
    if name == "transformer":
        return PatchTransformer(c, k, L, d_model=g("hidden", 64), layers=g("layers", 3))
    raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
