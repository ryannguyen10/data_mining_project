"""
evaluate.py
-----------
Evaluation metrics for multi-class classification:
    - Macro-F1
    - Micro-F1
    - Hamming Loss
"""

import numpy as np
from sklearn.metrics import f1_score, hamming_loss as sk_hamming_loss


def evaluate_classification(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             n_classes: int = None) -> dict:
    """Compute Macro-F1, Micro-F1, and Hamming Loss.

    Parameters
    ----------
    y_true    : true class labels [N]
    y_pred    : predicted class labels [N]
    n_classes : total number of classes (used for label list)

    Returns
    -------
    dict with keys: macro_f1, micro_f1, hamming_loss
    """
    labels = list(range(n_classes)) if n_classes else None

    macro_f1 = f1_score(y_true, y_pred,
                        average="macro",
                        labels=labels,
                        zero_division=0)
    micro_f1 = f1_score(y_true, y_pred,
                        average="micro",
                        labels=labels,
                        zero_division=0)
    hamming = sk_hamming_loss(y_true, y_pred)

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "hamming_loss": float(hamming),
    }


def per_class_f1(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 n_classes: int) -> np.ndarray:
    """Return F1 score for each class individually."""
    labels = list(range(n_classes))
    scores = f1_score(y_true, y_pred,
                      average=None,
                      labels=labels,
                      zero_division=0)
    return scores


def print_report(y_true: np.ndarray, y_pred: np.ndarray,
                 n_classes: int, title: str = "Evaluation"):
    """Pretty-print a full evaluation report."""
    metrics = evaluate_classification(y_true, y_pred, n_classes)
    per_class = per_class_f1(y_true, y_pred, n_classes)

    width = 50
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")
    print(f"  Macro-F1     : {metrics['macro_f1']:.4f}")
    print(f"  Micro-F1     : {metrics['micro_f1']:.4f}")
    print(f"  Hamming Loss : {metrics['hamming_loss']:.4f}")
    print(f"{'─'*width}")
    print(f"  Per-class F1:")
    for c, score in enumerate(per_class):
        count = (y_true == c).sum()
        print(f"    Class {c:2d}  |  F1={score:.4f}  |  support={count:5d}")
    print(f"{'═'*width}\n")
    return metrics
