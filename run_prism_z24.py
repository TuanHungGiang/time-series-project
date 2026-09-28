"""Train PRISM (full 6000-sample sequences, no compression) on the processed Z24 dataset.

Sample index = scenario*90 + setup*10 + segment (17 scenarios x 9 setups x 10 segments),
so splits are done by group instead of randomly to avoid leakage between segments cut
from the same long recording.

  --split segment : per (scenario, setup) recording, segments 0-6 train / 7 val / 8-9 test
                    (same recordings in every split; optimistic)
  --split setup   : setups 0-5 train / 6 val / 7-8 test (unseen measurement setups; strict)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "PRISM"))
from models.PRISM import Model  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--split", choices=["segment", "setup"], default="setup")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
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
idx = np.arange(N)
setup = (idx % 90) // 10
segment = idx % 10

if args.split == "segment":
    tr, va, te = segment <= 6, segment == 7, segment >= 8
else:
    tr, va, te = setup <= 5, setup == 6, setup >= 7
tr, va, te = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
print(f"split={args.split} train={len(tr)} val={len(va)} test={len(te)}", flush=True)

# per-channel standardisation with train statistics only
print("[2/4] standardising per channel (train statistics) ...", flush=True)
mu = X[tr].mean(axis=(0, 2), keepdims=True)
sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-8
X = ((X - mu) / sd).astype(np.float32)
Xt = torch.from_numpy(X)
yt = torch.from_numpy(y)

cfg = SimpleNamespace(task_name="classification", seq_len=X.shape[2], enc_in=X.shape[1],
                      num_class=int(y.max()) + 1, d_model=128, dropout=0.1)
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


def batches(ids, bs, shuffle):
    ids = np.random.permutation(ids) if shuffle else ids
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        # model expects (B, L, C)
        yield Xt[b].transpose(1, 2).to(dev, non_blocking=True), yt[b].to(dev)


@torch.no_grad()
def evaluate(ids):
    net.eval()
    loss, preds = 0.0, []
    for xb, yb in batches(ids, 32, False):
        out = net(xb)
        loss += crit(out, yb).item() * len(yb)
        preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return loss / len(ids), float((preds == y[ids]).mean()), preds


out_dir = Path(args.out)
out_dir.mkdir(exist_ok=True)
tag = f"prism_{args.split}_seed{args.seed}"
best = {"val_loss": np.inf}
bad = 0
n_batches = -(-len(tr) // args.batch_size)  # ceil
if args.max_train_batches:
    n_batches = min(n_batches, args.max_train_batches)
print(f"[4/4] training: up to {args.epochs} epochs, {n_batches} batches/epoch, "
      f"early-stop patience {args.patience}", flush=True)
t_start = time.time()
for ep in range(args.epochs):
    lr = args.lr * (0.5 ** ep)  # PRISM's 'type1' schedule: halve every epoch
    for g in opt.param_groups:
        g["lr"] = lr
    net.train()
    t0, tl, nb = time.time(), 0.0, 0
    for xb, yb in batches(tr, args.batch_size, True):
        opt.zero_grad()
        loss = crit(net(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        opt.step()
        tl += loss.item()
        nb += 1
        if nb % args.log_every == 0 or nb == n_batches:
            el = time.time() - t0
            print(f"  ep {ep+1:3d} | batch {nb:3d}/{n_batches} ({100*nb/n_batches:3.0f}%) "
                  f"| loss {tl/nb:.4f} | {el:.0f}s elapsed, ~{el/nb*(n_batches-nb):.0f}s left in epoch",
                  flush=True)
        if args.max_train_batches and nb >= args.max_train_batches:
            break
    print(f"  ep {ep+1:3d} | evaluating val/test ...", flush=True)
    vl, va_acc, _ = evaluate(va)
    tl_, te_acc, _ = evaluate(te)
    total = time.time() - t_start
    print(f"ep {ep+1:3d}/{args.epochs} lr {lr:.1e} train_loss {tl/nb:.4f} val_loss {vl:.4f} "
          f"val_acc {va_acc:.4f} test_acc {te_acc:.4f} | epoch {time.time()-t0:.0f}s, "
          f"total {total/60:.1f} min, no-improve {bad}/{args.patience}", flush=True)
    if vl < best["val_loss"]:
        best = {"val_loss": vl, "epoch": ep + 1, "val_acc": va_acc}
        torch.save(model.state_dict(), out_dir / f"{tag}.pt")
        bad = 0
        print(f"  -> new best val_loss {vl:.4f}, checkpoint saved", flush=True)
    else:
        bad += 1
        if bad >= args.patience:
            print(f"early stop at epoch {ep+1} (best epoch {best['epoch']})", flush=True)
            break

# final test at the checkpoint with the best validation loss
model.load_state_dict(torch.load(out_dir / f"{tag}.pt"))
tl_, te_acc, preds = evaluate(te)
from sklearn.metrics import f1_score, confusion_matrix  # noqa: E402

f1 = float(f1_score(y[te], preds, average="macro"))
cm = confusion_matrix(y[te], preds, labels=list(range(cfg.num_class)))
res = {"split": args.split, "seed": args.seed, "params": n_params, "best_epoch": best["epoch"],
       "val_acc": best["val_acc"], "test_acc": te_acc, "test_macro_f1": f1,
       "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
print(json.dumps(res), flush=True)
np.save(out_dir / f"{tag}_confusion.npy", cm)
(out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2))
