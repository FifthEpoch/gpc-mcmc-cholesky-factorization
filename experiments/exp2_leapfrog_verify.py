"""
Verify whether low HMC autocorrelation estimates in exp2 leapfrog sweeps are reliable.

This script intentionally reuses exp2.run_hmc rather than reimplementing the
sampler. It runs sampler diagnostics only; predictive metrics are out of scope.

Examples:
    python experiments/exp2_leapfrog_verify.py \
        --factor-dir data/rpchol_smoke \
        --test-embeddings datasets/pcam_test_embedding.npy \
        --test-labels datasets/pcam_test_label.npy \
        --k 100 \
        --output-dir data/exp2_leapfrog_verify_smoke \
        --exp3-per-dim-tau \
        --exp3-n-samples 500

    python experiments/exp2_leapfrog_verify.py \
        --factor-dir data/rpchol_full \
        --test-embeddings datasets/pcam_test_embedding.npy \
        --test-labels datasets/pcam_test_label.npy \
        --k 200 \
        --output-dir data/exp2_leapfrog_verify_full \
        --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from experiments.exp2 import (  # type: ignore[import-not-found]  # noqa: E402
        grad_log_posterior_with_cache,
        log_posterior,
        run_hmc,
        sample_predictive_probabilities_pivots,
    )
except ModuleNotFoundError:
    from exp2 import (  # type: ignore[import-not-found]  # noqa: E402
        grad_log_posterior_with_cache,
        log_posterior,
        run_hmc,
        sample_predictive_probabilities_pivots,
    )

# Keep imported names visible to static checkers and future readers; the diagnostic
# currently only calls run_hmc, but these imports intentionally verify exp2's public
# sampler/predictive helpers remain importable from this sibling script.
_EXP2_HELPERS = (
    log_posterior,
    grad_log_posterior_with_cache,
    sample_predictive_probabilities_pivots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify HMC tau/R-hat diagnostics for exp2 leapfrog settings."
    )
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--test-embeddings", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--n-leapfrog", type=int, default=35)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hmc-step-constant", type=float, default=0.05)
    parser.add_argument("--hmc-target-accept", type=float, default=0.8)
    parser.add_argument(
        "--no-adapt-step-size",
        action="store_false",
        dest="adapt_step_size",
        help="Disable HMC dual averaging step-size adaptation during warmup.",
    )

    parser.add_argument("--exp1-long-chain", action="store_true")
    parser.add_argument("--exp1-n-samples", type=int, default=5000)
    parser.add_argument("--exp1-n-warmup", type=int, default=1000)

    parser.add_argument("--exp2-multichain", action="store_true")
    parser.add_argument("--exp2-n-chains", type=int, default=4)
    parser.add_argument("--exp2-n-samples", type=int, default=1000)
    parser.add_argument("--exp2-n-warmup", type=int, default=500)

    parser.add_argument("--exp3-per-dim-tau", action="store_true")
    parser.add_argument(
        "--exp3-reuse-hmc",
        type=str,
        default=None,
        help="If provided, load nu_samples from this npz instead of running HMC.",
    )
    parser.add_argument("--exp3-n-samples", type=int, default=2000)
    parser.add_argument("--exp3-n-warmup", type=int, default=500)

    parser.add_argument("--all", action="store_true", help="Run all three experiments.")
    return parser.parse_args()


def load_training_state(factor_dir: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    factor_path = factor_dir / f"factor_k{k}.npy"
    labels_path = factor_dir / "labels.npy"
    if not factor_path.exists():
        raise FileNotFoundError(f"Missing factor file: {factor_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    F = np.load(factor_path).astype(np.float64, copy=False)
    y = np.load(labels_path).astype(np.float32, copy=False).squeeze()
    if F.shape[0] != y.shape[0]:
        raise ValueError(f"F/y row mismatch: F={F.shape}, y={y.shape}")
    return F, y


def run_hmc_diagnostic(
    F: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    n_leapfrog: int,
    hmc_step_constant: float,
    target_accept: float,
    adapt_step_size: bool,
    label: str,
) -> dict[str, Any]:
    initial_step = hmc_step_constant / (F.shape[1] ** 0.25)
    print(
        f"\nRunning {label}: n_samples={n_samples}, n_warmup={n_warmup}, "
        f"n_leapfrog={n_leapfrog}, seed={seed}, initial_step={initial_step:.6f}"
    )
    t0 = time.perf_counter()
    stats = run_hmc(
        factor=F,
        y=y,
        n_samples=n_samples,
        n_warmup=n_warmup,
        seed=seed,
        initial_step_size=initial_step,
        n_leapfrog=n_leapfrog,
        target_accept=target_accept,
        adapt_step_size=adapt_step_size,
    )
    wall = time.perf_counter() - t0
    stats["wall_time"] = float(wall)
    print(
        f"  done: wall={wall:.2f}s, accept={stats['accept_rate']:.3f}, "
        f"final_step={stats['step_size']:.6f}"
    )
    return stats


def integrated_time_vector(chain: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return per-dimension tau and whether the chain is shorter than 50*tau.

    emcee's integrated_time treats a 2D array as (step, walker), which is not
    what we want for one HMC chain with many nu dimensions. This uses the same
    FFT/autowindow idea on each scalar trace independently.
    """
    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 1:
        chain = chain[:, np.newaxis]
    elif chain.ndim == 2:
        pass
    else:
        chain = chain.reshape(chain.shape[0], -1)

    tau = np.array([integrated_time_1d(chain[:, j]) for j in range(chain.shape[1])])
    warned = bool(np.any(chain.shape[0] < 50.0 * tau))
    return tau, warned


def autocorr_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    if n < 2 or np.allclose(x, 0.0):
        return np.ones(n, dtype=float)

    n_fft = 1 << (2 * n - 1).bit_length()
    fft = np.fft.fft(x, n=n_fft)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n].real
    acf /= acf[0]
    return acf


def auto_window(taus: np.ndarray, c: float = 5.0) -> int:
    m = np.arange(taus.size) < c * taus
    if np.any(~m):
        return int(np.argmin(m))
    return int(taus.size - 1)


def integrated_time_1d(x: np.ndarray) -> float:
    acf = autocorr_1d(x)
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = auto_window(taus)
    # HMC can produce negative lag-one correlation, yielding formal tau < 1.
    # For this diagnostic, report the conservative ESS convention tau >= 1.
    return float(max(taus[window], 1.0))


def tau_summary(chain: np.ndarray) -> dict[str, Any]:
    tau_vec, warned = integrated_time_vector(chain)
    return {
        "tau_vec": tau_vec,
        "tau_mean": float(np.nanmean(tau_vec)),
        "tau_max": float(np.nanmax(tau_vec)),
        "tau_median": float(np.nanmedian(tau_vec)),
        "emcee_warning": bool(warned),
    }


def tau_scalar(chain: np.ndarray) -> tuple[float, bool]:
    tau_vec, warned = integrated_time_vector(chain)
    if tau_vec.size == 0:
        return float("nan"), True
    return float(np.nanmean(tau_vec)), warned


def experiment_1_long_chain(F: np.ndarray, y: np.ndarray, args: argparse.Namespace, out_dir: Path) -> dict:
    stats = run_hmc_diagnostic(
        F,
        y,
        n_samples=args.exp1_n_samples,
        n_warmup=args.exp1_n_warmup,
        seed=args.seed,
        n_leapfrog=args.n_leapfrog,
        hmc_step_constant=args.hmc_step_constant,
        target_accept=args.hmc_target_accept,
        adapt_step_size=args.adapt_step_size,
        label="Experiment 1 long chain",
    )

    requested_lengths = [500, 1000, 2000, 3000, 5000]
    lengths = [n for n in requested_lengths if n <= stats["nu_samples"].shape[0]]
    tau_rows = []

    print("\nExperiment 1: chain-length tau stability")
    print("chain_length  tau_nu_mean  tau_nu_max  tau_nu_median  tau_logp  emcee_warning")
    for n in lengths:
        nu_summary = tau_summary(stats["nu_samples"][:n])
        tau_logp, logp_warned = tau_scalar(stats["logp_trace"][:n, None])
        emcee_warning = bool(
            nu_summary["emcee_warning"]
            or logp_warned
            or not (n >= 50.0 * nu_summary["tau_mean"])
        )
        tau_rows.append(
            (
                n,
                nu_summary["tau_mean"],
                nu_summary["tau_max"],
                nu_summary["tau_median"],
                tau_logp,
                emcee_warning,
            )
        )
        print(
            f"{n:12d}  {nu_summary['tau_mean']:11.2f}  {nu_summary['tau_max']:10.2f}  "
            f"{nu_summary['tau_median']:13.2f}  {tau_logp:8.2f}  "
            f"{'yes' if emcee_warning else 'no':>13s}"
        )

    tau_table = np.array(
        tau_rows,
        dtype=[
            ("chain_length", "i8"),
            ("tau_nu_mean", "f8"),
            ("tau_nu_max", "f8"),
            ("tau_nu_median", "f8"),
            ("tau_logp", "f8"),
            ("emcee_warning", "?"),
        ],
    )
    np.savez(
        out_dir / "exp1_long_chain_results.npz",
        tau_table=tau_table,
        nu_samples=stats["nu_samples"],
        logp_trace=stats["logp_trace"],
        accept_rate=stats["accept_rate"],
        step_size=stats["step_size"],
        wall_time=stats["wall_time"],
    )

    first = float(tau_table["tau_nu_mean"][0]) if tau_table.size else float("nan")
    last = float(tau_table["tau_nu_mean"][-1]) if tau_table.size else float("nan")
    return {"first_tau": first, "last_tau": last, "tau_table": tau_table}


def basic_rhat(chains: np.ndarray) -> np.ndarray:
    """Basic Gelman-Rubin R-hat over shape (n_chains, n_draws, ...)."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape[0], chains.shape[1]
    flat = chains.reshape(m, n, -1)
    chain_means = np.mean(flat, axis=1)
    chain_vars = np.var(flat, axis=1, ddof=1)
    W = np.mean(chain_vars, axis=0)
    B = n * np.var(chain_means, axis=0, ddof=1)
    var_hat = ((n - 1.0) / n) * W + B / n
    rhat = np.sqrt(var_hat / W)
    return rhat.reshape(chains.shape[2:])


def arviz_rhat_or_none(chains: np.ndarray, name: str) -> np.ndarray | None:
    try:
        import arviz as az

        dataset = az.convert_to_dataset({name: chains})
        values = np.asarray(az.rhat(dataset)[name], dtype=float)
        return values
    except Exception:
        return None


def experiment_2_multichain(F: np.ndarray, y: np.ndarray, args: argparse.Namespace, out_dir: Path) -> dict:
    chains = []
    logp_chains = []
    accept_rates = []
    step_sizes = []

    print(
        f"\nMulti-chain R-hat diagnostic ({args.exp2_n_chains} chains, "
        f"n_leapfrog={args.n_leapfrog}, n_samples={args.exp2_n_samples} each)"
    )
    for chain_idx in range(args.exp2_n_chains):
        seed = args.seed + chain_idx
        stats = run_hmc_diagnostic(
            F,
            y,
            n_samples=args.exp2_n_samples,
            n_warmup=args.exp2_n_warmup,
            seed=seed,
            n_leapfrog=args.n_leapfrog,
            hmc_step_constant=args.hmc_step_constant,
            target_accept=args.hmc_target_accept,
            adapt_step_size=args.adapt_step_size,
            label=f"Experiment 2 chain {chain_idx + 1}/{args.exp2_n_chains}",
        )
        chains.append(stats["nu_samples"])
        logp_chains.append(stats["logp_trace"])
        accept_rates.append(float(stats["accept_rate"]))
        step_sizes.append(float(stats["step_size"]))

    chains_arr = np.stack(chains, axis=0)
    logp_arr = np.stack(logp_chains, axis=0)

    rhat_nu = arviz_rhat_or_none(chains_arr, "nu")
    rhat_logp_arr = arviz_rhat_or_none(logp_arr, "logp")
    rhat_method = "arviz rank-normalized split R-hat"
    if rhat_nu is None or rhat_logp_arr is None:
        rhat_nu = basic_rhat(chains_arr)
        rhat_logp_arr = basic_rhat(logp_arr[:, :, None])
        rhat_method = "basic Gelman-Rubin R-hat"

    rhat_logp = float(np.ravel(rhat_logp_arr)[0])
    rhat_nu_max = float(np.nanmax(rhat_nu))
    rhat_nu_mean = float(np.nanmean(rhat_nu))

    print(f"  R-hat method:       {rhat_method}")
    print(f"  R-hat (nu) max:    {rhat_nu_max:.3f}   (target: < 1.01)")
    print(f"  R-hat (nu) mean:   {rhat_nu_mean:.3f}")
    print(f"  R-hat (logp):      {rhat_logp:.3f}")
    print(f"  Per-chain accept_rates: {[round(x, 3) for x in accept_rates]}")
    print(f"  Per-chain final step sizes: {[round(x, 6) for x in step_sizes]}")

    if rhat_nu_max < 1.01:
        interpretation = "PASS: chains agree"
    elif rhat_nu_max <= 1.05:
        interpretation = "MARGINAL: borderline convergence"
    else:
        interpretation = "FAIL: chains disagree"
    print(f"  {interpretation}")

    np.savez(
        out_dir / "exp2_multichain_results.npz",
        chains=chains_arr,
        logp_chains=logp_arr,
        rhat_nu=rhat_nu,
        rhat_logp=rhat_logp,
        accept_rates=np.asarray(accept_rates),
        step_sizes=np.asarray(step_sizes),
        rhat_method=rhat_method,
    )

    return {
        "rhat_nu_max": rhat_nu_max,
        "rhat_nu_mean": rhat_nu_mean,
        "rhat_logp": rhat_logp,
        "interpretation": interpretation,
    }


def load_reused_nu_samples(path: Path, rank: int) -> np.ndarray:
    cached = np.load(path)
    if "nu_samples" not in cached:
        raise KeyError(f"{path} does not contain 'nu_samples'")
    nu_samples = np.asarray(cached["nu_samples"], dtype=np.float64)
    if nu_samples.ndim != 2 or nu_samples.shape[1] != rank:
        raise ValueError(
            f"Cached nu_samples has shape {nu_samples.shape}; expected second dimension {rank}"
        )
    print(f"Loaded cached nu_samples from {path}: shape={nu_samples.shape}")
    return nu_samples


def experiment_3_per_dim_tau(F: np.ndarray, y: np.ndarray, args: argparse.Namespace, out_dir: Path) -> dict:
    if args.exp3_reuse_hmc is not None:
        nu_samples = load_reused_nu_samples(Path(args.exp3_reuse_hmc), rank=F.shape[1])
    else:
        stats = run_hmc_diagnostic(
            F,
            y,
            n_samples=args.exp3_n_samples,
            n_warmup=args.exp3_n_warmup,
            seed=args.seed,
            n_leapfrog=args.n_leapfrog,
            hmc_step_constant=args.hmc_step_constant,
            target_accept=args.hmc_target_accept,
            adapt_step_size=args.adapt_step_size,
            label="Experiment 3 per-dim tau",
        )
        nu_samples = stats["nu_samples"]

    tau_vec, warned = integrated_time_vector(nu_samples)
    sorted_idx = np.argsort(tau_vec)[::-1]
    tau_mean = float(np.nanmean(tau_vec))
    tau_median = float(np.nanmedian(tau_vec))
    tau_max = float(np.nanmax(tau_vec))
    tau_q90 = float(np.nanquantile(tau_vec, 0.90))
    tau_q99 = float(np.nanquantile(tau_vec, 0.99))

    print(f"\nPer-dimension tau_nu analysis (k={F.shape[1]} dimensions)")
    print(f"  mean:    {tau_mean:.2f}")
    print(f"  median:  {tau_median:.2f}")
    print(f"  max:     {tau_max:.2f}")
    print(f"  q90:     {tau_q90:.2f}")
    print(f"  q99:     {tau_q99:.2f}")
    print(f"  emcee warning: {'yes' if warned else 'no'}")
    print("\n  Top 10 slowest dimensions:")
    print("    dim   tau")
    for idx in sorted_idx[:10]:
        print(f"    {int(idx):3d}   {tau_vec[idx]:.2f}")

    np.save(out_dir / "per_dim_tau.npy", tau_vec)
    np.savez(
        out_dir / "exp3_per_dim_tau_results.npz",
        per_dim_tau=tau_vec,
        sorted_indices=sorted_idx,
        tau_mean=tau_mean,
        tau_median=tau_median,
        tau_max=tau_max,
        tau_q90=tau_q90,
        tau_q99=tau_q99,
        emcee_warning=warned,
    )

    return {
        "tau_mean": tau_mean,
        "tau_median": tau_median,
        "tau_max": tau_max,
        "tau_ratio": tau_max / tau_mean if tau_mean > 0 else float("nan"),
    }


def interpret_exp1(first_tau: float, last_tau: float) -> str:
    if not np.isfinite(first_tau) or not np.isfinite(last_tau) or first_tau <= 0:
        return "unavailable"
    ratio = last_tau / first_tau
    if 0.8 <= ratio <= 1.2:
        return "stable"
    if ratio <= 1.8:
        return "mild bias"
    return "strong bias"


def interpret_exp2(rhat_max: float) -> str:
    if not np.isfinite(rhat_max):
        return "unavailable"
    if rhat_max < 1.01:
        return "chains agree"
    if rhat_max <= 1.05:
        return "borderline"
    return "disagree"


def interpret_exp3(tau_ratio: float) -> str:
    if not np.isfinite(tau_ratio):
        return "unavailable"
    if tau_ratio <= 3.0:
        return "no slow direction"
    if tau_ratio <= 5.0:
        return "mild"
    return "hidden slow direction"


def print_final_summary(results: dict[str, dict]) -> None:
    print("\n=== Verification Summary ===")
    overall_flags = []

    if "exp1" in results:
        first_tau = results["exp1"]["first_tau"]
        last_tau = results["exp1"]["last_tau"]
        interp = interpret_exp1(first_tau, last_tau)
        overall_flags.append(interp)
        print(
            f"Exp 1 (chain length stability):  tau_nu changed from {first_tau:.2f} "
            f"(n=first) to {last_tau:.2f} (n=last)"
        )
        print(f"  -> {interp}")
    else:
        print("Exp 1 (chain length stability):  not run")

    if "exp2" in results:
        rhat = results["exp2"]["rhat_nu_max"]
        interp = interpret_exp2(rhat)
        overall_flags.append(interp)
        print(f"Exp 2 (multi-chain agreement):   R-hat max = {rhat:.3f}")
        print(f"  -> {interp}")
    else:
        print("Exp 2 (multi-chain agreement):   not run")

    if "exp3" in results:
        tau_max = results["exp3"]["tau_max"]
        tau_mean = results["exp3"]["tau_mean"]
        tau_ratio = results["exp3"]["tau_ratio"]
        interp = interpret_exp3(tau_ratio)
        overall_flags.append(interp)
        print(
            f"Exp 3 (per-dimension tau):       tau_nu max = {tau_max:.2f} "
            f"vs mean = {tau_mean:.2f} (ratio: {tau_ratio:.1f})"
        )
        print(f"  -> {interp}")
    else:
        print("Exp 3 (per-dimension tau):       not run")

    if any(flag in {"strong bias", "disagree", "hidden slow direction"} for flag in overall_flags):
        overall = "sampling has problems"
    elif any(flag in {"mild bias", "borderline", "mild"} for flag in overall_flags):
        overall = "tau may be biased"
    elif overall_flags:
        overall = "tau values are reliable"
    else:
        overall = "no experiments were run"
    print(f"\nOverall: {overall}")


def main() -> None:
    args = parse_args()
    if not (
        args.all
        or args.exp1_long_chain
        or args.exp2_multichain
        or args.exp3_per_dim_tau
    ):
        print("No experiment selected. Use --exp1-long-chain, --exp2-multichain, --exp3-per-dim-tau, or --all.\n")
        print("Run with --help to see examples and options.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    F, y = load_training_state(Path(args.factor_dir), args.k)
    print(f"Loaded training state: F={F.shape} dtype={F.dtype}, y={y.shape}")
    print(f"Test arguments are accepted for CLI consistency but not used by these sampler diagnostics.")

    results: dict[str, dict] = {}
    if args.all or args.exp1_long_chain:
        results["exp1"] = experiment_1_long_chain(F, y, args, out_dir)
    if args.all or args.exp2_multichain:
        results["exp2"] = experiment_2_multichain(F, y, args, out_dir)
    if args.all or args.exp3_per_dim_tau:
        results["exp3"] = experiment_3_per_dim_tau(F, y, args, out_dir)

    print_final_summary(results)


if __name__ == "__main__":
    main()
