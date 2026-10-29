"""Common evaluation artifacts, plus explicit empty-case semantics."""

import numpy as np
from .serialization import write_json


def mask_metrics(truth, prediction):
    a, b = np.asarray(truth), np.asarray(prediction)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Masks must share a 2D shape")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Masks must be binary")
    a, b = a.astype(bool), b.astype(bool)
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    total = int(a.sum() + b.sum())
    return dict(
        dice=2 * intersection / total if total else 1.0,
        iou=intersection / union if union else 1.0,
        empty_truth=not bool(a.any()),
        empty_prediction=not bool(b.any()),
    )


def classification_metrics(truth, probabilities, threshold=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

    y = np.asarray(truth)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matched binary truth and probability vectors")
    if (
        not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid probabilities or threshold")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(
        n=len(y),
        threshold=threshold,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        sensitivity=float(tp / (tp + fn)) if tp + fn else None,
        specificity=float(tn / (tn + fp)) if tn + fp else None,
        auroc=float(roc_auc_score(y, p)) if both else None,
        average_precision=float(average_precision_score(y, p)) if both else None,
    )


def save_evaluation(path, task, split, metrics, provenance):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation requires val or test")
    write_json(
        path,
        dict(
            schema_version="1.0",
            task=task,
            split=split,
            metrics=metrics,
            provenance=provenance,
            clinically_validated=False,
        ),
    )
