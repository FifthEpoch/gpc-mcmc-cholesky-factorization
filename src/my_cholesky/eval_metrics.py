"""
Shared evaluation metrics for Experiments 1--3.

Provides AUROC, AUPRC, ECE, Brier score, sensitivity / false-negative rate,
and a reliability-diagram plotter.  All functions accept plain NumPy arrays
so they work identically for GP posteriors and NN predictions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve (binary)."""
    return float(roc_auc_score(y_true, y_prob))


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve (binary)."""
    return float(average_precision_score(y_true, y_prob))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of predicted probabilities."""
    return float(brier_score_loss(y_true, y_prob))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error with equal-width bins.

    Bins predicted probabilities into *n_bins* uniform intervals and returns
    the weighted-average |accuracy - confidence| gap.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += np.sum(mask) / n * abs(bin_acc - bin_conf)
    return float(ece)


def compute_classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Accuracy, sensitivity (recall), and false-negative rate at *threshold*.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "number_errors": int(fp + fn),
        "sensitivity": sensitivity,
        "sensitivity_TPR": sensitivity,
        "sensitivity_tpr": sensitivity,
        "specificity_TNR": specificity,
        "specificity_tnr": specificity,
        "false_negative_rate": 1.0 - sensitivity,
        "FNR": 1.0 - sensitivity,
        "false_positive_rate": 1.0 - specificity,
        "FPR": 1.0 - specificity,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Convenience wrapper that returns every metric in a single dict."""
    try:
        from predictive_metrics import evaluate_binary_probabilistic_predictions

        return evaluate_binary_probabilistic_predictions(
            y_true=y_true,
            p_pred=y_prob,
            threshold=threshold,
            n_bins=n_bins,
        )
    except Exception:
        pass

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    log_pred_prob_i = y_true * np.log(y_prob) + (1 - y_true) * np.log(1.0 - y_prob)
    elpd = float(np.sum(log_pred_prob_i))
    elpd_mean = float(np.mean(log_pred_prob_i))
    metrics = {
        "auroc": compute_auroc(y_true, y_prob),
        "auprc": compute_auprc(y_true, y_prob),
        "brier": compute_brier(y_true, y_prob),
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
        "elpd": elpd,
        "elpd_mean": elpd_mean,
        "pell": float("nan"),
        "pell_mean": float("nan"),
        "log_likelihood_mean": -float(log_loss(y_true, y_prob, labels=[0, 1])),
        "negative_log_likelihood_mean": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "predictive_likelihood": float(np.exp(elpd_mean)),
    }
    metrics.update(compute_classification_report(y_true, y_prob, threshold))
    return metrics


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    ax: Optional[plt.Axes] = None,
    title: str = "Reliability Diagram",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a reliability (calibration) diagram.

    Returns the (fig, ax) pair so the caller can save or display it.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not np.any(mask):
            bin_accs.append(np.nan)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(float(np.mean(y_true[mask])))
            bin_confs.append(float(np.mean(y_prob[mask])))
            bin_counts.append(int(np.sum(mask)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(midpoints, bin_accs, width=1.0 / n_bins, alpha=0.5, edgecolor="k", label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
