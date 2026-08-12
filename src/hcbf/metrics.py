from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    auroc: float
    brier: float
    ece: float
    precision_class0: float
    recall_class0: float
    f1_class0: float
    precision_class1: float
    recall_class1: float
    f1_class1: float
    class0_to_class1_error: float
    class1_to_class0_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        left, right = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    ece_bins: int = 15,
) -> BinaryMetrics:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must be one-dimensional and aligned.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite values in [0,1].")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must contain only 0 and 1.")

    y_pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    n0 = int(matrix[0].sum())
    n1 = int(matrix[1].sum())
    error_0_to_1 = float(matrix[0, 1] / n0) if n0 else float("nan")
    error_1_to_0 = float(matrix[1, 0] / n1) if n1 else float("nan")
    auroc = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else float("nan")

    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        auroc=auroc,
        brier=float(brier_score_loss(y_true, probabilities)),
        ece=expected_calibration_error(y_true, probabilities, n_bins=ece_bins),
        precision_class0=float(precision[0]),
        recall_class0=float(recall[0]),
        f1_class0=float(f1[0]),
        precision_class1=float(precision[1]),
        recall_class1=float(recall[1]),
        f1_class1=float(f1[1]),
        class0_to_class1_error=error_0_to_1,
        class1_to_class0_error=error_1_to_0,
    )
