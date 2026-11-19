import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score, roc_curve


def publication_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Record-window metrics with damage as the positive class.

    AUC is ROC-AUC and AP is average precision (area under the step-wise
    precision-recall curve). No point adjustment is applied.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = scores >= float(threshold)
    healthy, damage = labels == 0, labels == 1
    recall = float(prediction[damage].mean())
    specificity = float((~prediction[healthy]).mean())
    fpr = 1.0 - specificity
    precision = float(precision_score(labels, prediction, zero_division=0))
    f1 = float(f1_score(labels, prediction, zero_division=0))
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    ba = 0.5 * (recall + specificity)
    return {
        "auc": auc,
        "ap": ap,
        "ba": ba,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        # Backward-compatible aliases used by previous analysis notebooks.
        "auroc": auc,
        "balanced_accuracy": ba,
    }


def recall_at_fpr(labels, scores, limit):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return float(tpr[fpr <= limit + 1e-12].max())


def robust_score_reference(scores: np.ndarray) -> tuple[float, float]:
    values = np.log1p(np.maximum(np.asarray(scores, float), 0.0))
    center = float(np.median(values))
    scale = float(max(1.4826 * np.median(np.abs(values - center)), 0.25 * np.std(values), 1e-8))
    return center, scale


def threshold_from_healthy(scores: np.ndarray, specificity: float) -> float:
    try:
        boundary = float(np.quantile(scores, specificity, method="higher"))
    except TypeError:
        boundary = float(np.quantile(scores, specificity, interpolation="higher"))
    # Prediction uses score >= threshold. Move one representable float above
    # the selected healthy boundary so ties at the boundary remain healthy and
    # the realized calibration specificity is at least the requested value.
    return float(np.nextafter(boundary, np.inf))
