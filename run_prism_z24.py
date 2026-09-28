"""Train PRISM (full 6000-sample sequences, no compression) on the processed Z24 dataset.

Data: (1530, 27, 6000). Each original 60000-sample recording was already cut into 10 segments of 6000
samples, so sample index = scenario*90 + setup*10 + segment (17 scenarios x 9 setups x 10 segments).
Splits are made by group instead of randomly to avoid leakage between segments of one recording:

  --split segment : per (scenario, setup) recording, segments 0-6 train / 7 val / 8-9 test
                    (same recordings in every split; optimistic)
  --split setup   : setups 0-5 train / 6 val / 7-8 test (unseen measurement setups; strict)

Pipeline: normalise per channel with train statistics -> augment the TRAIN set only (on the fly, every epoch) ->
train PRISM -> report accuracy / precision / recall / F1 / ROC-AUC and save figures (loss curves, confusion
matrix, ROC, t-SNE 2D and 3D) for the checkpoint with the best validation loss.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # for report.py, independent of how python was launched
sys.path.insert(0, str(ROOT / "PRISM"))
from models.PRISM import Model  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--split", choices=["segment", "setup"], default="setup")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lr_schedule", choices=["halve", "constant", "cosine"], default="halve",
               help="halve = PRISM default (lr/2 every epoch); constant; cosine decay over --epochs")
p.add_argument("--norm", choices=["robust", "zscore"], default="robust",
               help="robust: per-channel median / median per-sample std (+clip); zscore: global mean/std")
p.add_argument("--clip", type=float, default=10.0, help="clip normalised values to +-clip (robust norm only; 0 = off)")
p.add_argument("--no_augment", action="store_true", help="disable training-set augmentation")
p.add_argument("--aug_noise", type=float, default=0.1, help="Gaussian noise std (signal std is ~1 after normalisation)")
p.add_argument("--aug_warp", type=float, default=0.05, help="max time-warp factor (+-); warping also shifts frequencies")
p.add_argument("--aug_crop_min", type=float, default=0.7, help="min fraction kept by random cropping")
p.add_argument("--no_plots", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_train_batches", type=int, default=0, help="debug: limit batches per epoch")
p.add_argument("--log_every", type=int, default=10, help="print training progress every N batches")
p.add_argument("--gpus", type=int, default=0,
               help="number of GPUs to use with DataParallel (0 = all visible GPUs)")
p.add_argument("--out", default=str(ROOT / "results"))
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = ROOT / "Z24-dataset-processed"
print("[1/4] loading inputs.npy (~1 GB) ...", flush=True)
X = np.load(data_dir / "inputs.npy")  # (1530, 27, 6000) float32
y = np.load(data_dir / "labels.npy").astype(np.int64)
N = len(y)
n_cls = int(y.max()) + 1
idx = np.arange(N)
setup = (idx % 90) // 10
segment = idx % 10

if args.split == "segment":
    tr, va, te = segment <= 6, segment == 7, segment >= 8
else:
    tr, va, te = setup <= 5, setup == 6, setup >= 7
tr, va, te = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
print(f"split={args.split} train={len(tr)} val={len(va)} test={len(te)}", flush=True)

# normalisation with TRAIN statistics only
print(f"[2/4] normalising per channel ({args.norm}, train statistics) ...", flush=True)
if args.norm == "zscore":
    mu = X[tr].mean(axis=(0, 2), keepdims=True)
    sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-8
else:
    # a few huge spikes inflate the global std ~10x, so scale by the median per-sample std instead
    Xtr = X[tr]
    mu = np.median(Xtr[:, :, ::10], axis=(0, 2))[None, :, None].astype(np.float32)
    sd = (np.median(Xtr.std(axis=2), axis=0)[None, :, None] + 1e-12).astype(np.float32)
    del Xtr
X = (X - mu) / sd
if args.norm == "robust" and args.clip > 0:
    np.clip(X, -args.clip, args.clip, out=X)
X = X.astype(np.float32, copy=False)
print(f"      typical per-sample/channel std after normalisation: {np.median(X[tr[:200]].std(axis=2)):.3f}", flush=True)
Xt = torch.from_numpy(X)
yt = torch.from_numpy(y)

cfg = SimpleNamespace(task_name="classification", seq_len=X.shape[2], enc_in=X.shape[1],
                      num_class=n_cls, d_model=128, dropout=0.1)
model = Model(cfg).to(dev)
n_params = sum(p.numel() for p in model.parameters())
n_gpu = torch.cuda.device_count() if dev.type == "cuda" else 0
if args.gpus:
    n_gpu = min(args.gpus, n_gpu)
# DataParallel splits each batch across GPUs; `model` stays the unwrapped module so checkpoints have clean keys
net = nn.DataParallel(model, device_ids=list(range(n_gpu))) if n_gpu > 1 else model
gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)) if n_gpu else "cpu"
print(f"[3/4] PRISM params: {n_params/1e6:.2f}M  device={dev}  gpus={max(n_gpu, 1)} ({gpu_names})  "
      f"batch {args.batch_size} total -> {args.batch_size // max(n_gpu, 1)} per GPU", flush=True)

opt = torch.optim.RAdam(model.parameters(), lr=args.lr)
crit = nn.CrossEntropyLoss()

# ---- training-set augmentation: each sample gets exactly one transform, drawn with these probabilities ----
AUG_NAMES = ["noise", "reverse", "crop", "warp", "shift"]
aug_p = torch.tensor([0.4, 0.3, 0.1, 0.1, 0.1], device=dev)


def augment(x):
    """x: (B, C, L) on `dev`. noise 40% / time reversal 30% / crop 10% / time warp 10% / shift 10%."""
    B, C, L = x.shape
    kind = torch.multinomial(aug_p, B, replacement=True)
    out = x.clone()
    ar = torch.arange(L, device=x.device)
    for k in range(len(AUG_NAMES)):
        m = (kind == k).nonzero(as_tuple=True)[0]
        if m.numel() == 0:
            continue
        xs, n = x[m], m.numel()
        if k == 0:    # additive Gaussian noise
            xs = xs + args.aug_noise * torch.randn_like(xs)
        elif k == 1:  # time reversal
            xs = xs.flip(-1)
        elif k == 2:  # random crop: keep one random window, zero the rest (frequency content preserved)
            length = (torch.empty(n, device=x.device).uniform_(args.aug_crop_min, 1.0) * L).long()
            start = (torch.rand(n, device=x.device) * (L - length + 1).float()).long()
            keep = (ar[None, :] >= start[:, None]) & (ar[None, :] < (start + length)[:, None])
            xs = xs * keep[:, None, :]
        elif k == 3:  # time warp: resample the time axis by a factor in [1-w, 1+w]
            s = torch.empty(n, device=x.device).uniform_(1 - args.aug_warp, 1 + args.aug_warp)
            pos = (ar[None, :].float() * s[:, None]).clamp(0, L - 1)
            i0 = pos.floor().long()
            i1 = (i0 + 1).clamp(max=L - 1)
            w = (pos - i0.float())[:, None, :]
            g0 = torch.gather(xs, 2, i0[:, None, :].expand(-1, C, -1))
            g1 = torch.gather(xs, 2, i1[:, None, :].expand(-1, C, -1))
            xs = (1 - w) * g0 + w * g1
        else:         # circular time shift by up to +-25% of the length
            shift = torch.randint(-(L // 4), L // 4 + 1, (n,), device=x.device)
            src = (ar[None, :] - shift[:, None]) % L
            xs = torch.gather(xs, 2, src[:, None, :].expand(-1, C, -1))
        out[m] = xs
    return out


def batches(ids, bs, shuffle):
    ids = np.random.permutation(ids) if shuffle else ids
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        yield Xt[b].to(dev, non_blocking=True), yt[b].to(dev)  # (B, C, L), labels


@torch.no_grad()
def evaluate(ids):
    net.eval()
    loss, preds, probs = 0.0, [], []
    for xb, yb in batches(ids, 32, False):
        out = net(xb.transpose(1, 2))  # model expects (B, L, C)
        loss += crit(out, yb).item() * len(yb)
        preds.append(out.argmax(1).cpu())
        probs.append(torch.softmax(out.float(), 1).cpu())
    preds, probs = torch.cat(preds).numpy(), torch.cat(probs).numpy()
    return loss / len(ids), float((preds == y[ids]).mean()), preds, probs


@torch.no_grad()
def embed(ids):
    """128-d PRISM embedding (channel-averaged features before the linear classifier)."""
    model.eval()
    return torch.cat([model.front(xb).mean(dim=1).cpu() for xb, _ in batches(ids, 32, False)]).numpy()


tag = f"prism_{args.split}_{args.lr_schedule}_{'noaug' if args.no_augment else 'aug'}_seed{args.seed}"
run_dir = Path(args.out) / tag
run_dir.mkdir(parents=True, exist_ok=True)
best = {"val_loss": np.inf}
bad = 0
hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
n_batches = -(-len(tr) // args.batch_size)  # ceil
if args.max_train_batches:
    n_batches = min(n_batches, args.max_train_batches)
print(f"[4/4] training: up to {args.epochs} epochs, {n_batches} batches/epoch, early-stop patience {args.patience}, "
      f"lr schedule {args.lr_schedule}, augmentation {'OFF' if args.no_augment else 'ON (train only)'}", flush=True)
t_start = time.time()
for ep in range(args.epochs):
    if args.lr_schedule == "constant":
        lr = args.lr
    elif args.lr_schedule == "cosine":
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * ep / args.epochs))
    else:
        lr = args.lr * (0.5 ** ep)  # PRISM's 'type1' schedule: halve every epoch
    for g in opt.param_groups:
        g["lr"] = lr
    net.train()
    t0, tl, nb, correct, seen = time.time(), 0.0, 0, 0, 0
    for xb, yb in batches(tr, args.batch_size, True):
        if not args.no_augment:
            xb = augment(xb)
        opt.zero_grad()
        out = net(xb.transpose(1, 2))
        loss = crit(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        opt.step()
        tl += loss.item()
        nb += 1
        correct += (out.argmax(1) == yb).sum().item()
        seen += len(yb)
        if nb % args.log_every == 0 or nb == n_batches:
            el = time.time() - t0
            print(f"  ep {ep+1:3d} | batch {nb:3d}/{n_batches} ({100*nb/n_batches:3.0f}%) "
                  f"| loss {tl/nb:.4f} | {el:.0f}s elapsed, ~{el/nb*(n_batches-nb):.0f}s left in epoch",
                  flush=True)
        if args.max_train_batches and nb >= args.max_train_batches:
            break
    print(f"  ep {ep+1:3d} | evaluating val/test ...", flush=True)
    vl, va_acc, _, _ = evaluate(va)
    _, te_acc, _, _ = evaluate(te)  # printed for monitoring only; never used to pick the checkpoint
    hist["train_loss"].append(tl / nb)
    hist["train_acc"].append(correct / seen)
    hist["val_loss"].append(vl)
    hist["val_acc"].append(va_acc)
    hist["lr"].append(lr)
    total = time.time() - t_start
    print(f"ep {ep+1:3d}/{args.epochs} lr {lr:.1e} train_loss {tl/nb:.4f} train_acc {correct/seen:.4f} "
          f"val_loss {vl:.4f} val_acc {va_acc:.4f} test_acc {te_acc:.4f} | epoch {time.time()-t0:.0f}s, "
          f"total {total/60:.1f} min, no-improve {bad}/{args.patience}", flush=True)
    if vl < best["val_loss"]:
        best = {"val_loss": vl, "epoch": ep + 1, "val_acc": va_acc}
        torch.save(model.state_dict(), run_dir / "best.pt")
        bad = 0
        print(f"  -> new best val_loss {vl:.4f}, checkpoint saved", flush=True)
    else:
        bad += 1
        if bad >= args.patience:
            print(f"early stop at epoch {ep+1} (best epoch {best['epoch']})", flush=True)
            break

# ---------------- final report on the checkpoint with the best validation loss ----------------
from report import compute_metrics, plot_confusion, plot_roc, plot_tsne, plot_training_curves  # noqa: E402

model.load_state_dict(torch.load(run_dir / "best.pt"))
_, _, val_pred, val_prob = evaluate(va)
_, _, te_pred, te_prob = evaluate(te)
m_val = compute_metrics(y[va], val_pred, val_prob, n_cls)
m_te = compute_metrics(y[te], te_pred, te_prob, n_cls)
res = {"split": args.split, "lr_schedule": args.lr_schedule, "augmentation": not args.no_augment, "norm": args.norm,
       "batch_size": args.batch_size, "seed": args.seed, "params": n_params, "best_epoch": best["epoch"],
       "epochs_run": len(hist["train_loss"]), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
       "val": {k: v for k, v in m_val.items() if k != "per_class"},
       "test": {k: v for k, v in m_te.items() if k != "per_class"},
       "test_per_class": m_te["per_class"]}
(run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
(run_dir / "history.json").write_text(json.dumps(hist, indent=2))
np.savez(run_dir / "predictions_test.npz", y_true=y[te], y_pred=te_pred, prob=te_prob, setup=setup[te])

print("\n================ TEST SET (best-val checkpoint) ================", flush=True)
for k, v in res["test"].items():
    print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}", flush=True)

if not args.no_plots:
    print("saving figures ...", flush=True)
    plot_training_curves(hist, run_dir / "training_curves.png")
    plot_confusion(y[te], te_pred, n_cls, run_dir / "confusion_matrix.png", title="Test set -")
    plot_roc(y[te], te_prob, n_cls, run_dir / "roc_curve.png")
    emb = embed(te)
    np.savez(run_dir / "embeddings_test.npz", emb=emb, y=y[te], setup=setup[te])
    plot_tsne(emb, y[te], setup[te], run_dir, seed=args.seed)
print(f"done. everything is in {run_dir}", flush=True)
