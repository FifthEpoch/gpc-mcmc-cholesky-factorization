import numpy as np
from scipy.special import log_expit, logsumexp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


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


def evaluate_binary_probabilistic_predictions(
    y_true,
    p_pred,
    threshold=0.5,
    n_bins=15,
    latent_samples=None,
):
    """Compute point-prediction metrics, plus Bayesian ELPD if latent samples are provided."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, 1e-12, 1.0 - 1e-12)

    y_hat = (p_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    number_errors = int(fp + fn)
    accuracy = float((tp + tn) / len(y_true))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan

    nll = float(log_loss(y_true, p_pred, labels=[0, 1]))
    log_likelihood = -nll

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
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "number_errors": number_errors,
        "sensitivity_TPR": sensitivity,
        "FNR": fnr,
        "specificity_TNR": specificity,
        "FPR": fpr,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "log_likelihood_mean": log_likelihood,
        "negative_log_likelihood_mean": nll,
    }

    if latent_samples is not None:
        latent_samples = np.asarray(latent_samples, dtype=float)
        if latent_samples.ndim != 2:
            raise ValueError("latent_samples must have shape (n_samples, n_test)")
        if latent_samples.shape[1] != y_true.shape[0]:
            raise ValueError(
                "latent_samples second dimension must match number of test points: "
                f"got {latent_samples.shape[1]} and {y_true.shape[0]}"
            )

        log_p_y = np.where(
            y_true[None, :] == 1,
            log_expit(latent_samples),
            log_expit(-latent_samples),
        )

        posterior_expected_log_likelihood = float(np.mean(log_p_y))
        metrics["posterior_expected_log_likelihood"] = posterior_expected_log_likelihood
        metrics["posterior_expected_nll"] = -posterior_expected_log_likelihood

        # For Bernoulli classification, log E[p(y | f*)] is identical to the
        # log-likelihood of the posterior mean probability because
        # E[1 - p] = 1 - E[p]. The posterior-expected log-likelihood above is
        # the genuinely different uncertainty-sensitive Jensen counterpart.
        elpd_i = logsumexp(log_p_y, axis=0) - np.log(latent_samples.shape[0])
        metrics["elpd_total"] = float(np.sum(elpd_i))
        metrics["elpd_per_point"] = float(np.mean(elpd_i))

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
