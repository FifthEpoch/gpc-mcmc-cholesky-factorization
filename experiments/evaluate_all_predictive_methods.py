import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predictive_metrics import compare_against_reference, print_metric_table


def load_array_file(path):
    if path.endswith(".npz"):
        return dict(np.load(path, allow_pickle=True))
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        if isinstance(data, dict):
            return data
        raise ValueError(f"Unsupported .npy object type: {type(data)} for file {path}")
    raise ValueError(f"Unsupported file type: {path}")


def first_available(*values):
    for value in values:
        if value is not None:
            return value
    return None


def get_summary_arrays(result):
    prob_mean = first_available(
        result.get("predictive_prob_test"),
        result.get("predictive_prob"),
        result.get("prob_test"),
        result.get("prob_mean"),
    )

    latent_mean = first_available(
        result.get("predictive_latent_mean_test"),
        result.get("predictive_latent_mean"),
        result.get("latent_mean_test"),
        result.get("latent_mean"),
    )

    latent_variance = first_available(
        result.get("predictive_latent_var_test"),
        result.get("predictive_latent_var"),
        result.get("latent_var_test"),
        result.get("latent_var"),
    )
    if latent_variance is None and result.get("predictive_latent_std") is not None:
        latent_variance = np.square(result["predictive_latent_std"])
    if latent_variance is None and result.get("predictive_std") is not None and latent_mean is not None:
        latent_variance = np.square(result["predictive_std"])

    return {
        "prob_mean": prob_mean,
        "latent_mean": latent_mean,
        "latent_variance": latent_variance,
        "prob_q05": result.get("prob_q05"),
        "prob_q50": result.get("prob_q50"),
        "prob_q95": result.get("prob_q95"),
        "latent_q05": result.get("latent_q05"),
        "latent_q50": result.get("latent_q50"),
        "latent_q95": result.get("latent_q95"),
    }


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")

    sources = {
        "exact_hmc": os.path.join(data_dir, "exp1_predictive_hmc_results.npz"),
        "low_rank_hmc": os.path.join(data_dir, "exp1_predictive_hmc_lowrank_nugget_results.npz"),
        "svgp": os.path.join(data_dir, "exp1_gpytorch_svgp_results.npy"),
        "pygp_la_ep": os.path.join(data_dir, "exp1_pygp_la_ep_results.npy"),
    }

    results = {}
    for name, path in sources.items():
        if os.path.exists(path):
            results[name] = load_array_file(path)
        else:
            print(f"Warning: missing results file for {name}: {path}")

    if "exact_hmc" not in results:
        raise RuntimeError("Reference exact HMC results are required for comparison.")

    reference = results["exact_hmc"]
    ref_summary = get_summary_arrays(reference)
    if ref_summary["prob_mean"] is None:
        raise RuntimeError(
            "Reference exact HMC results are required for comparison and must include predictive probability arrays."
        )

    for name in ["low_rank_hmc", "svgp"]:
        if name not in results:
            continue
        candidate = results[name]
        candidate_summary = get_summary_arrays(candidate)

        if candidate_summary["prob_mean"] is None:
            print(f"Skipping comparison for {name}: missing predictive probability summaries.")
            continue

        try:
            comparison = compare_against_reference(ref_summary, candidate_summary)
        except ValueError as exc:
            print(f"Skipping comparison for {name}: {exc}")
            continue
        print_metric_table(comparison, title=f"Comparison vs exact HMC: {name}")

    if "pygp_la_ep" in results:
        candidate = results["pygp_la_ep"]
        for method in candidate.get("methods", []):
            method_data = candidate.get("results", {}).get(method, {})
            candidate_summary = get_summary_arrays(method_data)
            if candidate_summary["prob_mean"] is None:
                print(f"Skipping comparison for pyGPs {method}: missing predictive probability summaries.")
                continue

            try:
                comparison = compare_against_reference(ref_summary, candidate_summary)
            except ValueError as exc:
                print(f"Skipping comparison for pyGPs {method}: {exc}")
                continue
            print_metric_table(comparison, title=f"Comparison vs exact HMC: pyGPs {method}")

    print("Done.")


if __name__ == "__main__":
    main()
