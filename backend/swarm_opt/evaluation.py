"""Classification evaluation for the three specialist agents.

Every metric is reported next to the majority-class baseline for the same
split, per the Phase 2 requirement not to present a metric as evidence of
predictive usefulness without a baseline comparison. This module measures
software/statistical correctness of the fitted classifier; it says nothing
about whether the resulting signal is tradeable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassificationReport:
    split: str
    n: int
    classes: list[str]
    class_counts: dict[str, int]
    majority_class: str
    majority_baseline_accuracy: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    log_loss: float
    multiclass_brier: float
    confusion_matrix: list[list[int]]
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_predictions(
    split: str,
    classes: tuple[str, ...],
    y_true: pd.Series,
    y_pred: list[str],
    proba: np.ndarray,
) -> ClassificationReport:
    """`proba` is `(n_samples, len(classes))`, columns ordered as `classes`."""
    n = len(y_true)
    counts = y_true.value_counts()
    class_counts = {c: int(counts.get(c, 0)) for c in classes}
    majority_class = max(class_counts, key=class_counts.get)
    majority_baseline_accuracy = class_counts[majority_class] / n if n else float("nan")

    accuracy = float(np.mean(np.array(y_pred) == y_true.to_numpy())) if n else float("nan")
    note = ""

    present = sorted(set(y_true.unique()))
    if len(present) < 2:
        balanced_acc = float("nan")
        macro_f1 = weighted_f1 = float("nan")
        prec = rec = {c: float("nan") for c in classes}
        note = (
            f"only one class ({present}) observed in this split's labels; "
            "balanced accuracy and F1 are undefined and reported as NaN"
        )
    else:
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0))
        weighted_f1 = float(
            f1_score(y_true, y_pred, labels=list(classes), average="weighted", zero_division=0)
        )
        p, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(classes), zero_division=0
        )
        prec = {c: float(v) for c, v in zip(classes, p)}
        rec = {c: float(v) for c, v in zip(classes, r)}

    y_idx = y_true.map({c: i for i, c in enumerate(classes)}).to_numpy()
    # Defensive renormalisation: AgentPrediction already validates each row
    # sums to 1 within 1e-3, but float32 XGBoost output can be a few ulps off
    # that sklearn's own (tighter) log_loss tolerance, which otherwise warns.
    row_sums = proba.sum(axis=1, keepdims=True)
    proba_norm = np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sums > 0)
    try:
        ll = float(log_loss(y_idx, proba_norm, labels=list(range(len(classes)))))
    except ValueError:
        ll = float("nan")

    onehot = np.zeros_like(proba_norm)
    onehot[np.arange(n), y_idx] = 1.0
    brier = float(np.mean(np.sum((proba_norm - onehot) ** 2, axis=1))) if n else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(classes)).tolist()

    return ClassificationReport(
        split=split,
        n=n,
        classes=list(classes),
        class_counts=class_counts,
        majority_class=majority_class,
        majority_baseline_accuracy=majority_baseline_accuracy,
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_precision=prec,
        per_class_recall=rec,
        log_loss=ll,
        multiclass_brier=brier,
        confusion_matrix=cm,
        note=note,
    )
