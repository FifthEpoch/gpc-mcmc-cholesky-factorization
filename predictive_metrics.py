import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


EPS = 1e-12


def expected_calibration_error(y_true, p_pred, n_bins=15):
    """ECE for binary probabilistic predictions."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]

        if b == n_bins - 1:
            in_bin = (p_pred >= lo) & (p_pred <= hi)
        else:
            in_bin = (p_pred >= lo) & (p_pred < hi)

        if np.any(in_bin):
            bin_confidence = np.mean(p_pred[in_bin])
            bin_accuracy = np.mean(y_true[in_bin])
            bin_weight = np.mean(in_bin)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def binary_log_predictive_terms(y_true, p_pred):
    """Pointwise log p(y_i | x_i, D) for binary predictive probabilities."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, EPS, 1.0 - EPS)
    return y_true * np.log(p_pred) + (1 - y_true) * np.log(1.0 - p_pred)


def posterior_expected_log_likelihood(y_true, p_samples):
    """
    PELL from posterior predictive probability samples.

    Returns the posterior expectation of total log likelihood:
    E_s[sum_i log p(y_i | theta_s, x_i)]. By Jensen's inequality this is <=
    the marginal ELPD computed from the posterior-mean predictive probability.
    """
    y_true = np.asarray(y_true).astype(int).reshape(1, -1)
    p_samples = np.asarray(p_samples).astype(float)
    if p_samples.ndim != 2:
        raise ValueError("p_samples must have shape (n_samples, n_observations)")
    if p_samples.shape[1] != y_true.shape[1]:
        raise ValueError(
            "p_samples second dimension must match len(y_true): "
            f"{p_samples.shape[1]} != {y_true.shape[1]}"
        )
    p_samples = np.clip(p_samples, EPS, 1.0 - EPS)
    log_lik_by_sample = np.sum(
        y_true * np.log(p_samples) + (1 - y_true) * np.log(1.0 - p_samples),
        axis=1,
    )
    pell = float(np.mean(log_lik_by_sample))
    return {
        "pell": pell,
        "pell_mean": float(pell / y_true.shape[1]),
        "posterior_expected_log_loss": float(-pell / y_true.shape[1]),
        "posterior_log_loss_mean": float(-pell / y_true.shape[1]),
        "posterior_total_log_loss": float(-pell),
    }


def evaluate_binary_probabilistic_predictions(
    y_true,
    p_pred,
    threshold=0.5,
    n_bins=15,
    p_samples=None,
):
    """Compute probabilistic and classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, EPS, 1.0 - EPS)

    y_hat = (p_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    number_errors = int(fp + fn)
    accuracy = float((tp + tn) / len(y_true))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan

    # Mean negative log predictive probability
    nll = float(log_loss(y_true, p_pred, labels=[0, 1]))

    # Mean log predictive probability
    log_likelihood = -nll

    log_pred_prob_i = binary_log_predictive_terms(y_true, p_pred)

    # ELPD: expected log predictive density over the dataset.
    # For classification this is the sum of log predictive probabilities.
    elpd = float(np.sum(log_pred_prob_i))
    elpd_mean = float(np.mean(log_pred_prob_i))

    brier = float(brier_score_loss(y_true, p_pred))
    ece = expected_calibration_error(y_true, p_pred, n_bins=n_bins)

    try:
        auroc = float(roc_auc_score(y_true, p_pred))
    except ValueError:
        auroc = np.nan

    try:
        auprc = float(average_precision_score(y_true, p_pred))
    except ValueError:
        auprc = np.nan

    metrics = {
        "log_likelihood_mean": log_likelihood,
        "negative_log_likelihood_mean": nll,

        "elpd": elpd,
        "elpd_mean": elpd_mean,

        "pell": np.nan,
        "pell_mean": np.nan,
        "mean_predictive_log_likelihood": elpd_mean,
        "predictive_likelihood": float(np.exp(elpd_mean)),

        "brier": brier,
        "ece": ece,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "number_errors": number_errors,
        "sensitivity_TPR": sensitivity,
        "sensitivity_tpr": sensitivity,
        "FNR": fnr,
        "false_negative_rate": fnr,
        "specificity_TNR": specificity,
        "specificity_tnr": specificity,
        "FPR": fpr,
        "false_positive_rate": fpr,
        "TP": int(tp),
        "tp": int(tp),
        "FP": int(fp),
        "fp": int(fp),
        "TN": int(tn),
        "tn": int(tn),
        "FN": int(fn),
        "fn": int(fn),
    }
    if p_samples is not None:
        metrics.update(posterior_expected_log_likelihood(y_true, p_samples))
    return metrics


def summarize_predictive_distribution(
    p_samples,
    latent_samples,
):
    """Summarize posterior predictive distribution."""
    p_samples = np.asarray(p_samples)
    latent_samples = np.asarray(latent_samples)

    summary = {
        "prob_mean": np.mean(p_samples, axis=0),
        "prob_variance": np.var(p_samples, axis=0, ddof=1),
        "prob_std": np.std(p_samples, axis=0, ddof=1),
        "prob_q05": np.quantile(p_samples, 0.05, axis=0),
        "prob_q50": np.quantile(p_samples, 0.50, axis=0),
        "prob_q95": np.quantile(p_samples, 0.95, axis=0),
        "latent_mean": np.mean(latent_samples, axis=0),
        "latent_variance": np.var(latent_samples, axis=0, ddof=1),
        "latent_std": np.std(latent_samples, axis=0, ddof=1),
        "latent_q05": np.quantile(latent_samples, 0.05, axis=0),
        "latent_q50": np.quantile(latent_samples, 0.50, axis=0),
        "latent_q95": np.quantile(latent_samples, 0.95, axis=0),
    }

    summary["prob_central_90_width"] = summary["prob_q95"] - summary["prob_q05"]
    summary["latent_central_90_width"] = summary["latent_q95"] - summary["latent_q05"]

    return summary


def print_metric_table(metrics, title="Metrics"):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.6f}")
        else:
            print(f"{key:32s}: {value}")


def print_posterior_statistics(
    latent_mean=None,
    latent_var=None,
    prob_mean=None,
    prob_var=None,
    title="Posterior statistics",
):
    """Print compact summaries for latent and predictive posterior arrays."""
    print(f"\n{title}")
    print("-" * len(title))

    values = {
        "latent_mean": latent_mean,
        "latent_var": latent_var,
        "prob_mean": prob_mean,
        "prob_var": prob_var,
    }
    for name, value in values.items():
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        print(
            f"{name:32s}: mean={np.nanmean(arr):.6f} "
            f"std={np.nanstd(arr):.6f} min={np.nanmin(arr):.6f} max={np.nanmax(arr):.6f}"
        )


def compare_against_reference(reference, candidate):
    out = {}
    for name in [
        "prob_mean",
        "prob_variance",
        "prob_q05",
        "prob_q50",
        "prob_q95",
        "latent_mean",
        "latent_variance",
        "latent_q05",
        "latent_q50",
        "latent_q95",
    ]:
        if name not in reference or name not in candidate:
            continue
        ref = reference[name]
        cand = candidate[name]
        if ref is None or cand is None:
            continue
        ref = np.asarray(ref)
        cand = np.asarray(cand)
        if ref.shape != cand.shape:
            continue
        out[f"{name}_mae"] = float(np.mean(np.abs(cand - ref)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean((cand - ref) ** 2)))

    if not out:
        raise ValueError("No comparable summary arrays were found between reference and candidate.")
    return out