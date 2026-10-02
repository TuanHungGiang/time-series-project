"""Train a classifier (PRISM or a sequence model from models_seq.py) on full 6000-sample sequences of the Z24 dataset.

Data: (1530, 27, 6000). Each original 60000-sample recording was already cut into 10 segments of 6000
samples, so sample index = scenario*90 + setup*10 + segment (17 scenarios x 9 setups x 10 segments).
Splits are made by group instead of randomly to avoid leakage between segments of one recording:

  --split segment : per (scenario, setup) recording, segments 0-6 train / 7 val / 8-9 test
                    (same recordings in every split; optimistic)
  --split setup           : setups 0-5 train / 6 val / 7-8 test (legacy strict split)
  --split setup_holdout   : setups 0-4 train / 5-6 val / 7-8 test (recommended hard split)
  --split low_data        : setups 0-2 train / 3-4 val / 5-8 test (low-data stress test)

For every split, validation/test predictions are aggregated by averaging the probabilities of the
segments that belong to the same original (scenario, setup) recording.  Recording-level metrics are
primary; segment-level metrics are saved as secondary diagnostics.

Pipeline: normalise per channel with train statistics -> augment the TRAIN set only (on the fly, every epoch) ->
train -> report accuracy / precision / recall / F1 / ROC-AUC and save figures (loss curves, confusion matrix,
ROC, t-SNE 2D and 3D) for the checkpoint with the best validation loss.

--train_window N additionally splits each sample along the TIME axis: train samples use only their first N
steps (then augmented), val/test samples use their remaining, later steps instead. --split still decides
which SAMPLES (recordings) are train/val/test, so this adds no cross-sample leakage; it tests whether
training on a short augmented snippet generalises to a longer, later portion of different recordings.
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
from models_seq import MODEL_NAMES, build_model  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", choices=MODEL_NAMES, default="prism",
               help="prism (CNN, reference) | ms4n (S4D state-space) | gru | lstm | cnn_lstm | transformer")
p.add_argument("--hidden", type=int, default=None, help="model width (default: 64)")
p.add_argument("--layers", type=int, default=None, help="depth (defaults: ms4n 1, gru/lstm 2, transformer 3)")
p.add_argument("--stem_stride", type=int, default=None,
               help="mamba/gru/lstm stride of the learned conv stem (model default if omitted)")
p.add_argument("--split", choices=["segment", "setup", "setup_holdout", "low_data"],
               default="setup_holdout")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight_decay", type=float, default=1e-4)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--lr_schedule", choices=["halve", "constant", "cosine"], default="halve",
               help="halve = PRISM default (lr/2 every epoch); constant; cosine decay over --epochs")
p.add_argument("--seq_len", type=int, default=0,
               help="use only this many time steps per sample (0 = all 6000); see --seq_mode")
p.add_argument("--seq_mode", choices=["crop", "downsample"], default="crop",
               help="crop: keep the first N points (same sample rate, shorter duration - the signal is cut "
                    "short but not distorted); downsample: N points evenly spaced across the full 6000 "
                    "(same duration, lower sample rate - aliases any frequency content above the new "
                    "Nyquist limit, i.e. it can distort the signal, not just shrink it)")
p.add_argument("--train_window", type=int, default=0,
               help="if >0, TRAIN samples use only their first N time steps (then augmented); VAL/TEST "
                    "samples use their remaining, later time steps instead (0 = disabled: everyone uses the "
                    "same, full-length signal). Which SAMPLES are train/val/test is still decided by --split, "
                    "so this does not add cross-sample leakage; it tests whether training on a short "
                    "augmented snippet generalises to a longer, later portion of different recordings.")
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
p.add_argument("--run-name", "--run_name", dest="run_name", default=None,
               help="custom results subdirectory name (default: generated from the configuration)")
args = p.parse_args()

if args.run_name is not None:
    run_name_path = Path(args.run_name)
    if not args.run_name.strip() or run_name_path.name != args.run_name or args.run_name in (".", ".."):
        p.error("--run-name must be one non-empty directory name without path separators")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = ROOT / "Z24-dataset-processed"
print("[1/4] loading inputs.npy (~1 GB) ...", flush=True)
X = np.load(data_dir / "inputs.npy")  # (1530, 27, 6000) float32
y = np.load(data_dir / "labels.npy").astype(np.int64)
if args.seq_len and args.seq_len < X.shape[2]:
    if args.seq_mode == "crop":
        X = X[:, :, :args.seq_len]
    else:
        t_idx = np.linspace(0, X.shape[2] - 1, args.seq_len).round().astype(int)
        X = X[:, :, t_idx]
    print(f"      seq_len: 6000 -> {X.shape[2]} ({args.seq_mode})", flush=True)
N = len(y)
n_cls = int(y.max()) + 1
idx = np.arange(N)
setup = (idx % 90) // 10
segment = idx % 10
recording = idx // 10

if args.split == "segment":
    tr, va, te = segment <= 6, segment == 7, segment >= 8
elif args.split == "setup":
    tr, va, te = setup <= 5, setup == 6, setup >= 7
elif args.split == "setup_holdout":
    tr, va, te = setup <= 4, (setup >= 5) & (setup <= 6), setup >= 7
else:  # low_data
    tr, va, te = setup <= 2, (setup >= 3) & (setup <= 4), setup >= 5
tr, va, te = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
if args.split != "segment":
    assert set(recording[tr]).isdisjoint(recording[va])
    assert set(recording[tr]).isdisjoint(recording[te])
    assert set(recording[va]).isdisjoint(recording[te])
print(f"split={args.split} segments: train={len(tr)} val={len(va)} test={len(te)} | "
      f"recordings: train={len(np.unique(recording[tr]))} "
      f"val={len(np.unique(recording[va]))} test={len(np.unique(recording[te]))}", flush=True)

# optional time-axis train/eval split: train samples see only the head, val/test samples only the tail
if args.train_window:
    tw = args.train_window
    assert 0 < tw < X.shape[2], f"--train_window must be in (0, {X.shape[2]})"
    X_train_src, X_eval_src = X[:, :, :tw], X[:, :, tw:]
    print(f"      train window: first {tw} steps (train, augmented) | "
          f"eval window: last {X.shape[2] - tw} steps (val/test, not augmented)", flush=True)
else:
    X_train_src = X_eval_src = X

# normalisation with TRAIN statistics only, computed from the exact slice the model trains on
print(f"[2/4] normalising per channel ({args.norm}, train statistics) ...", flush=True)
Xtr = X_train_src[tr]
if args.norm == "zscore":
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
else:
    # a few huge spikes inflate the global std ~10x, so scale by the median per-sample std instead
    mu = np.median(Xtr[:, :, ::10], axis=(0, 2))[None, :, None].astype(np.float32)
    sd = (np.median(Xtr.std(axis=2), axis=0)[None, :, None] + 1e-12).astype(np.float32)
del Xtr


def _norm(A):
    A = (A - mu) / sd
    if args.norm == "robust" and args.clip > 0:
        np.clip(A, -args.clip, args.clip, out=A)
    return A.astype(np.float32, copy=False)


X_train_src = _norm(X_train_src)
X_eval_src = X_train_src if X_eval_src is X else _norm(X_eval_src)  # avoid a redundant full-array copy
print(f"      typical per-sample/channel std after normalisation: "
      f"{np.median(X_train_src[tr[:200]].std(axis=2)):.3f}", flush=True)
Xt_train, Xt_eval = torch.from_numpy(X_train_src), torch.from_numpy(X_eval_src)
yt = torch.from_numpy(y)

cfg = SimpleNamespace(task_name="classification", seq_len=max(X_train_src.shape[2], X_eval_src.shape[2]),
                      enc_in=X.shape[1], num_class=n_cls, d_model=128, dropout=0.1)
model = build_model(args.model, cfg, args).to(dev)
n_params = sum(p.numel() for p in model.parameters())
n_gpu = torch.cuda.device_count() if dev.type == "cuda" else 0
if args.gpus:
    n_gpu = min(args.gpus, n_gpu)
# DataParallel splits each batch across GPUs; `model` stays the unwrapped module so checkpoints have clean keys
net = nn.DataParallel(model, device_ids=list(range(n_gpu))) if n_gpu > 1 else model
gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)) if n_gpu else "cpu"
print(f"[3/4] model {args.model}: {n_params/1e6:.3f}M params  device={dev}  gpus={max(n_gpu, 1)} ({gpu_names})  "
      f"batch {args.batch_size} total -> {args.batch_size // max(n_gpu, 1)} per GPU", flush=True)

opt = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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


def batches(ids, bs, shuffle, src):
    ids = np.random.permutation(ids) if shuffle else ids
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        yield src[b].to(dev, non_blocking=True), yt[b].to(dev)  # (B, C, L), labels


@torch.no_grad()
def evaluate(ids):
    """Always reads from Xt_eval: val/test samples only ever use their (possibly windowed) eval slice."""
    net.eval()
    loss, preds, probs = 0.0, [], []
    for xb, yb in batches(ids, 32, False, Xt_eval):
        out = net(xb.transpose(1, 2))  # model expects (B, L, C)
        loss += crit(out, yb).item() * len(yb)
        preds.append(out.argmax(1).cpu())
        probs.append(torch.softmax(out.float(), 1).cpu())
    preds, probs = torch.cat(preds).numpy(), torch.cat(probs).numpy()
    return loss / len(ids), float((preds == y[ids]).mean()), preds, probs


def aggregate_recordings(ids, probs):
    """Average segment probabilities into one prediction per source recording."""
    groups = recording[ids]
    rec_ids = np.unique(groups)
    rec_y, rec_prob, rec_setup, counts = [], [], [], []
    for rec_id in rec_ids:
        positions = np.where(groups == rec_id)[0]
        labels = np.unique(y[ids[positions]])
        setups = np.unique(setup[ids[positions]])
        if len(labels) != 1 or len(setups) != 1:
            raise ValueError(f"Recording {rec_id} has inconsistent labels or setups")
        probability = probs[positions].mean(axis=0, dtype=np.float64)
        probability /= probability.sum()
        rec_y.append(int(labels[0]))
        rec_prob.append(probability)
        rec_setup.append(int(setups[0]))
        counts.append(len(positions))
    rec_y = np.asarray(rec_y)
    rec_prob = np.asarray(rec_prob)
    rec_pred = rec_prob.argmax(axis=1)
    true_prob = rec_prob[np.arange(len(rec_y)), rec_y]
    return {
        "y": rec_y,
        "pred": rec_pred,
        "prob": rec_prob,
        "recording_ids": rec_ids,
        "setup": np.asarray(rec_setup),
        "segment_count": np.asarray(counts),
        "loss": float(-np.log(np.clip(true_prob, 1e-12, 1.0)).mean()),
        "accuracy": float((rec_pred == rec_y).mean()),
    }


@torch.no_grad()
def embed(ids):
    """Features right before the classifier head (used for t-SNE); reads from Xt_eval, see evaluate()."""
    model.eval()
    return torch.cat([model.embed(xb.transpose(1, 2)).cpu() for xb, _ in batches(ids, 32, False, Xt_eval)]).numpy()


tw_tag = f"_tw{args.train_window}" if args.train_window else ""
tag = (args.run_name or
       f"{args.model}_{args.split}_{args.lr_schedule}_{'noaug' if args.no_augment else 'aug'}{tw_tag}_seed{args.seed}")
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
    for xb, yb in batches(tr, args.batch_size, True, Xt_train):
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
    print(f"  ep {ep+1:3d} | evaluating validation recordings ...", flush=True)
    _, va_segment_acc, _, va_prob = evaluate(va)
    va_record = aggregate_recordings(va, va_prob)
    vl, va_acc = va_record["loss"], va_record["accuracy"]
    hist["train_loss"].append(tl / nb)
    hist["train_acc"].append(correct / seen)
    hist["val_loss"].append(vl)
    hist["val_acc"].append(va_acc)
    hist["lr"].append(lr)
    total = time.time() - t_start
    print(f"ep {ep+1:3d}/{args.epochs} lr {lr:.1e} train_loss {tl/nb:.4f} train_acc {correct/seen:.4f} "
          f"val_record_loss {vl:.4f} val_record_acc {va_acc:.4f} "
          f"val_segment_acc {va_segment_acc:.4f} | epoch {time.time()-t0:.0f}s, "
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

model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev, weights_only=True))
_, _, val_segment_pred, val_segment_prob = evaluate(va)
_, _, te_segment_pred, te_segment_prob = evaluate(te)
val_record = aggregate_recordings(va, val_segment_prob)
te_record = aggregate_recordings(te, te_segment_prob)
m_val = compute_metrics(val_record["y"], val_record["pred"], val_record["prob"], n_cls)
m_te = compute_metrics(te_record["y"], te_record["pred"], te_record["prob"], n_cls)
m_val_segment = compute_metrics(y[va], val_segment_pred, val_segment_prob, n_cls)
m_te_segment = compute_metrics(y[te], te_segment_pred, te_segment_prob, n_cls)
res = {"model": args.model, "split": args.split, "lr_schedule": args.lr_schedule, "augmentation": not args.no_augment, "norm": args.norm,
       "batch_size": args.batch_size, "seed": args.seed, "params": n_params, "best_epoch": best["epoch"],
       "epochs_run": len(hist["train_loss"]),
       "primary_evaluation_unit": "recording",
       "split_sizes_segments": {"train": len(tr), "validation": len(va), "test": len(te)},
       "split_sizes_recordings": {"train": len(np.unique(recording[tr])),
                                   "validation": len(val_record["y"]), "test": len(te_record["y"])},
       "val": {k: v for k, v in m_val.items() if k != "per_class"},
       "test": {k: v for k, v in m_te.items() if k != "per_class"},
       "test_per_class": m_te["per_class"],
       "val_segment": {k: v for k, v in m_val_segment.items() if k != "per_class"},
       "test_segment": {k: v for k, v in m_te_segment.items() if k != "per_class"}}
(run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
(run_dir / "history.json").write_text(json.dumps(hist, indent=2))
np.savez(run_dir / "predictions_test.npz", y_true=y[te], y_pred=te_segment_pred,
         prob=te_segment_prob, setup=setup[te], recording_ids=recording[te], segment=segment[te])
np.savez(run_dir / "predictions_test_recording.npz", y_true=te_record["y"],
         y_pred=te_record["pred"], prob=te_record["prob"], setup=te_record["setup"],
         recording_ids=te_record["recording_ids"], segment_count=te_record["segment_count"])

print("\n================ TEST SET (best-val checkpoint) ================", flush=True)
for k, v in res["test"].items():
    print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}", flush=True)

if not args.no_plots:
    print("saving figures ...", flush=True)
    plot_training_curves(hist, run_dir / "training_curves.png")
    plot_confusion(te_record["y"], te_record["pred"], n_cls,
                   run_dir / "confusion_matrix.png", title="Test recordings -")
    plot_roc(te_record["y"], te_record["prob"], n_cls, run_dir / "roc_curve.png")
    emb = embed(te)
    rec_emb = np.stack([emb[recording[te] == rec_id].mean(axis=0)
                        for rec_id in te_record["recording_ids"]])
    np.savez(run_dir / "embeddings_test.npz", emb=rec_emb, y=te_record["y"],
             setup=te_record["setup"], recording_ids=te_record["recording_ids"])
    plot_tsne(rec_emb, te_record["y"], te_record["setup"], run_dir, seed=args.seed)
print(f"done. everything is in {run_dir}", flush=True)
