# Experiment Results Form

Use this document plus the CSV templates in this folder to record **all Exp0-4 outcomes** in a plotting-friendly format. Use one row per experimental condition: for example, one row per `(method, k)` in Exp0, one row per `(sampler, k)` in Exp2, and one row per `(dataset, method, embedding_config)` in Exp3/4.

Machine-readable templates:

- `scripts/experiment_results_runs_template.csv`
- `scripts/experiment_results_template.csv`

Both CSV files use the same schema.

---

## Experiment Map

| Experiment | Main scripts | Row granularity | Primary outcomes |
|---|---|---|---|
| `exp0` | `exp0_algorithm_verification.py`, `exp0_algorithm_verification_fast_decay.py` | one row per kernel/method/k or kernel/method/N | approximation error and runtime |
| `exp1` | `exp1_mcmc_gpc.py`, `exp1_mcmc_gpc_2.py`, `exp1_predictive.py` | one row per method/k or predictive run | factor time, MCMC time, acceptance, predictive metrics |
| `exp2` | `exp1b_emcee_gpc.py`, `exp1b_mala.py`, proposed-method sweeps | one row per sampler/config/k | ESS, tau, acceptance, ELPD, runtime |
| `exp3` | `exp3_nn_baseline.py`, `exp3_unet_classifier.py` | one row per dataset/model/embedding config | AUROC, AUPRC, calibration, confusion rates, runtime |
| `exp4` | `exp4_tabpfn_baseline.py` | one row per dataset/embedding/TabPFN config | AUROC, AUPRC, ELPD, calibration, confusion rates, runtime |

---

## Common Metadata

Always fill these when possible:

| Field | Meaning |
|---|---|
| `record_id` | Unique row id, e.g. `wc3013-2026-04-26-exp2-hmc-k100-a` |
| `experiment` | `exp0`, `exp1`, `exp2`, `exp3`, or `exp4` |
| `script_path` | Exact script or sbatch used |
| `method_name` | Algorithm/model, e.g. `Accel`, `RPChol`, `SVGP`, `MLP`, `TabPFN` |
| `sampler` | MCMC sampler if applicable, e.g. `RWM`, `MALA`, `HMC`, `emcee-MALA` |
| `dataset` | `synthetic_blobs`, `synthetic_kernel`, `pcam`, `camelyon17`, `embed`, etc. |
| `job_id`, `account_partition`, `hostname`, `node_gpu` | Cluster execution details |
| `code_ref` | Git commit SHA |
| `run_timestamp_utc` | Timestamp in UTC |
| `notes` | Any deviations, failures, or manual details |

---

## Reproducibility Config Fields

Use the relevant fields for each experiment and leave non-applicable fields blank.

### Exp0: RPCholesky Algorithm Verification

Record: `kernel`, `synthetic_d`, `synthetic_bandwidth`, `fixed_n`, `n`, `k`, `k_list`, `trials`, `block_size_b`, `n_values`, `data_seed`.

Outcomes: `mean_time_sec`, `std_time_sec`, `mean_rel_trace_error`, `approx_error_fro_rel`, `artifacts`.

### Exp1: Dense vs RPCholesky GPC / Predictive Runs

Record: `data_generator`, `data_seed`, `n_train`, `n_test`, `kernel`, `kernel_bandwidth`, `n_samples`, `n_warmup`, `k`, `rp_k_values`, `arpcholesky_b`, `rwm_seed`, `jitter`, `step_size`, `n_leapfrog`.

Outcomes: `factor_time_sec`, `per_step_time_sec`, `total_mcmc_time_sec`, `accept_rate`, `final_step_size`, `approx_error_fro_rel`, `elpd`, `auroc`, `auprc`, `brier`, `ece`, `artifacts`.

### Exp2: Proposed Method and MCMC Hyperparameter Sweeps

Record every sweep parameter: `k`, `sampler`, `n_samples`, `n_warmup`, `n_walkers`, `step_size`, `n_leapfrog`, `arpcholesky_b`, `kernel`, `kernel_bandwidth`, `jitter`, `num_inducing`, `learning_rate`, `batch_size`, plus any custom hyperparameters in `notes`.

Outcomes: `elpd`, `ess_logp`, `ess_per_sec`, `tau`, `accept_rate`, `per_step_time_sec`, `total_mcmc_time_sec`, `fit_or_train_time_sec`, `inference_time_sec`, `auroc`, `auprc`, `brier`, `ece`, `false_negative_rate`, `total_pipeline_time_sec`.

### Exp3: Deterministic Baselines

Record: `dataset`, `model_architecture`, `encoder`, `embedding_source`, `embedding_root`, `embedding_variant`, `feature_dim`, `skip_embed`, `epochs`, `patience`, `batch_size`, `lr`, `weight_decay`, `hidden_dim`, `dropout`, `base_channels`, `image_size`, `num_workers`, `max_train_samples`, `max_valid_samples`, `max_test_samples`, `seed`.

Outcomes: `elpd`, `auroc`, `auprc`, `accuracy`, `precision`, `recall`, `sensitivity_tpr`, `specificity_tnr`, `false_positive_rate`, `false_negative_rate`, `brier`, `ece`, `tp`, `tn`, `fp`, `fn`, `fit_or_train_time_sec`, `inference_time_sec`, `total_pipeline_time_sec`.

### Exp4: TabPFN Baseline

Record: `dataset`, `embedding_source`, `embedding_root`, `embedding_variant`, `feature_dim`, `max_train_samples`, `predict_chunk_size`, `device`, `threshold`, `seed`, `tabpfn_version`.

Outcomes: `elpd`, `auroc`, `auprc`, `accuracy`, `sensitivity_tpr`, `specificity_tnr`, `false_positive_rate`, `false_negative_rate`, `brier`, `ece`, `tp`, `tn`, `fp`, `fn`, `fit_or_train_time_sec`, `inference_time_sec`.

---

## Core Metric Definitions

| Field | Meaning |
|---|---|
| `elpd` | Expected log pointwise predictive density on held-out/test data. Higher is better. Important for proposed-method sweeps. |
| `auroc`, `auprc` | Ranking performance metrics. Higher is better. |
| `brier`, `ece` | Probabilistic calibration metrics. Lower is better. |
| `tp`, `tn`, `fp`, `fn` | Confusion counts at `threshold`. |
| `false_negative_rate` | `fn / (tp + fn)`. Lower means fewer missed positives. |
| `false_positive_rate` | `fp / (fp + tn)`. |
| `ess_per_sec` | MCMC effective sample size per second. Higher is better. |
| `tau` | Integrated autocorrelation time. Lower is better. |
| `mean_rel_trace_error` | Exp0 trace approximation error. Lower is better. |
| `approx_error_fro_rel` | Relative Frobenius kernel approximation error. Lower is better. |

---

## Quality Checklist

- `record_id` is unique.
- Config fields match actual stdout/JSON/NPY/MAT values, not just intended values.
- Time columns are in seconds.
- MCMC time fields specify whether they are post-warmup only; current scripts generally report post-warmup MCMC time.
- Classification counts satisfy `tp + tn + fp + fn = n_test`.
- `false_negative_rate ~= fn / (tp + fn)`.
- For Exp3/4 with partner embeddings, verify `feature_dim` and row-wise label alignment.
