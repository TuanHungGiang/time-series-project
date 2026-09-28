"""Metrics and figures for the Z24 PRISM experiments (training curves, confusion matrix, ROC, t-SNE 2D/3D)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, cohen_kappa_score,  # noqa: E402
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, top_k_accuracy_score)
from sklearn.preprocessing import label_binarize  # noqa: E402


def class_colors(n):
    return plt.cm.tab20(np.linspace(0, 1, 20))[:n] if n <= 20 else plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))


def compute_metrics(y_true, y_pred, prob, n_cls):
    """Accuracy, precision, recall, F1 (macro and weighted), ROC-AUC, top-3, kappa, MCC."""
    prob = np.asarray(prob, dtype=np.float64)
    prob = prob / prob.sum(axis=1, keepdims=True)
    m = {"accuracy": float(accuracy_score(y_true, y_pred)),
         "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        m[f"precision_{avg}"], m[f"recall_{avg}"], m[f"f1_{avg}"] = float(pr), float(rc), float(f1)
    labels = list(range(n_cls))
    try:
        m["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=labels))
        m["roc_auc_ovr_weighted"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="weighted", labels=labels))
    except ValueError:
        m["roc_auc_ovr_macro"] = m["roc_auc_ovr_weighted"] = None
    m["top3_accuracy"] = float(top_k_accuracy_score(y_true, prob, k=3, labels=labels))
    m["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    m["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    m["per_class"] = {str(c): {"precision": float(pr[c]), "recall": float(rc[c]), "f1": float(f1[c]),
                               "support": int(sup[c])} for c in labels}
    return m


def plot_training_curves(hist, path):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(ep, hist["train_loss"], "o-", ms=3, label="train (augmented)")
    ax[0].plot(ep, hist["val_loss"], "s-", ms=3, label="validation")
    ax[0].set(title="Loss convergence", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(ep, hist["train_acc"], "o-", ms=3, label="train (augmented)")
    ax[1].plot(ep, hist["val_acc"], "s-", ms=3, label="validation")
    ax[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
    ax[2].semilogy(ep, hist["lr"], "o-", ms=3, color="tab:green")
    ax[2].set(title="Learning rate", xlabel="epoch", ylabel="lr")
    best = int(np.argmin(hist["val_loss"])) + 1
    for a in ax[:2]:
        a.axvline(best, color="gray", ls="--", lw=1, label=f"best epoch ({best})")
        a.grid(alpha=0.3)
        a.legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, n_cls, path, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(17, 7.5))
    for a, data, sub, fmt in ((ax[0], cm, "counts", "d"), (ax[1], cmn, "row-normalised (recall)", ".2f")):
        im = a.imshow(data, cmap="Blues", vmin=0)
        a.set(title=f"{title} {sub}".strip(), xlabel="predicted scenario", ylabel="true scenario",
              xticks=range(n_cls), yticks=range(n_cls))
        thr = data.max() / 2
        for i in range(n_cls):
            for j in range(n_cls):
                v = data[i, j]
                if v > 0:
                    a.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6.5,
                           color="white" if v > thr else "black")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_roc(y_true, prob, n_cls, path):
    prob = np.asarray(prob, dtype=np.float64)
    Y = label_binarize(y_true, classes=list(range(n_cls)))
    cols = class_colors(n_cls)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    tprs = []
    for c in range(n_cls):
        if Y[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(Y[:, c], prob[:, c])
        ax.plot(fpr, tpr, color=cols[c], lw=1, alpha=0.7, label=f"scenario {c} (AUC {auc(fpr, tpr):.2f})")
        tprs.append(np.interp(grid, fpr, tpr))
    macro = np.mean(tprs, axis=0)
    macro[0] = 0.0
    fpr_mi, tpr_mi, _ = roc_curve(Y.ravel(), prob.ravel())
    ax.plot(grid, macro, "k-", lw=2.5, label=f"macro-average (AUC {auc(grid, macro):.3f})")
    ax.plot(fpr_mi, tpr_mi, "k--", lw=2, label=f"micro-average (AUC {auc(fpr_mi, tpr_mi):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set(title="ROC curves (one-vs-rest)", xlabel="false positive rate", ylabel="true positive rate",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne(emb, y, setup_ids, out_dir, seed=0):
    """t-SNE of learned embeddings: 2D (coloured by scenario and by setup) and 3D (by scenario)."""
    n_cls = int(y.max()) + 1
    cols = class_colors(n_cls)
    perp = float(min(30, max(5, (len(y) - 1) // 3)))
    kw = dict(perplexity=perp, init="pca", learning_rate="auto", random_state=seed)
    z2 = TSNE(n_components=2, **kw).fit_transform(emb)
    z3 = TSNE(n_components=3, **kw).fit_transform(emb)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for c in range(n_cls):
        s = y == c
        ax[0].scatter(z2[s, 0], z2[s, 1], color=cols[c], s=22, label=f"scenario {c}", alpha=0.85)
    ax[0].set_title("t-SNE 2D - coloured by damage scenario")
    setups = np.unique(setup_ids)
    scol = plt.cm.tab10(np.arange(len(setups)) % 10)
    for k, sv in enumerate(setups):
        s = setup_ids == sv
        ax[1].scatter(z2[s, 0], z2[s, 1], color=scol[k], s=22, label=f"setup {sv}", alpha=0.85)
    ax[1].set_title("t-SNE 2D - coloured by measurement setup")
    for a in ax:
        a.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "tsne_2d.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(9.5, 8))
    ax3 = fig.add_subplot(projection="3d")
    for c in range(n_cls):
        s = y == c
        ax3.scatter(z3[s, 0], z3[s, 1], z3[s, 2], color=cols[c], s=20, label=f"scenario {c}", alpha=0.85)
    ax3.set(title="t-SNE 3D - coloured by damage scenario", xlabel="t-SNE 1", ylabel="t-SNE 2", zlabel="t-SNE 3")
    ax3.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "tsne_3d.png", dpi=150)
    plt.close(fig)
    np.savez(Path(out_dir) / "tsne_coords.npz", z2=z2, z3=z3, y=y, setup=setup_ids)
