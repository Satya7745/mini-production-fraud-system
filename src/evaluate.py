"""Threshold tuning, classification metrics, confidence intervals, and plots."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_threshold_by_cost(
    y_true: Iterable[int],
    scores: Iterable[float],
    false_negative_cost: float = 25.0,
    false_positive_cost: float = 1.0,
) -> dict[str, float]:
    """Choose a validation-only threshold minimizing C_fn*FN + C_fp*FP.

    The implementation sorts once and evaluates every distinct score in O(n log n),
    including the no-alert operating point.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    y_sorted, p_sorted = y[order], p[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    positives = int(tp[-1])
    distinct = np.r_[p_sorted[:-1] != p_sorted[1:], True]
    indices = np.flatnonzero(distinct)
    costs = fp[indices] * false_positive_cost + (positives - tp[indices]) * false_negative_cost
    thresholds = p_sorted[indices]

    no_alert_threshold = np.nextafter(float(p_sorted[0]), np.inf)
    costs = np.r_[positives * false_negative_cost, costs]
    thresholds = np.r_[no_alert_threshold, thresholds]
    best_index = int(np.argmin(costs))
    threshold = float(thresholds[best_index])
    prediction = (p >= threshold).astype(int)
    tn, fp_count, fn_count, tp_count = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "validation_cost": float(costs[best_index]),
        "tn": int(tn),
        "fp": int(fp_count),
        "fn": int(fn_count),
        "tp": int(tp_count),
        "false_negative_cost": float(false_negative_cost),
        "false_positive_cost": float(false_positive_cost),
    }


def classification_metrics(y_true: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    cost = fp + 25 * fn
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost": float(cost),
        "cost_per_1000": float(1000 * cost / len(y)),
        "alert_rate": float(prediction.mean()),
    }


def stratified_bootstrap_pr_auc(
    y_true: Iterable[int], scores: Iterable[float], n_bootstrap: int = 1000, seed: int = 42
) -> dict[str, float]:
    """Return a stratified percentile interval for test PR-AUC."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample_idx = np.r_[
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ]
        estimates[i] = average_precision_score(y[sample_idx], p[sample_idx])
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(average_precision_score(y, p)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "n_bootstrap": int(n_bootstrap),
        "method": "stratified percentile bootstrap",
    }


def save_precision_recall_plot(results: dict[str, tuple[np.ndarray, np.ndarray]], output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, (y, scores) in results.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    prevalence = float(next(iter(results.values()))[0].mean())
    ax.axhline(prevalence, color="#666666", linestyle="--", label=f"Random={prevalence:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Locked Test Precision-Recall Curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_confusion_plot(metrics: dict[str, float], output: str | Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{int(value):,}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Legitimate", "Fraud alert"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Locked Test Confusion Matrix (t={metrics['threshold']:.4g})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
