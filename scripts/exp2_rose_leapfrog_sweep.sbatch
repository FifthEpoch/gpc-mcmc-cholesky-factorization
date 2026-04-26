#!/bin/bash
#SBATCH --job-name=exp2_lf_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/exp2_rose/%x_%j.out
#SBATCH --error=slurm_logs/exp2_rose/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Submit with, for example:
#   sbatch --account=csci_ga_2565-2026sp --partition=<partition> \
#          --export=ALL,NETID=sd6701 scripts/exp2_rose_leapfrog_sweep.sbatch
#
# Default sweep:
#   n_leapfrog in {4, 6, 10, 16, 25}
#   k=200, n_samples=1000, n_warmup=500
#
# Optional overrides:
#   NETID, PROJECT_ROOT, CONDA_ENV
#   FACTOR_DIR             datasets/pcam-hg/train/embeddings/exp2_rpchol_full
#   EMBEDDINGS             optional; default is FACTOR_DIR/embeddings.npy
#   TEST_EMBEDDINGS        datasets/pcam-hg/test/embeddings/projected_512.npy
#   TEST_LABELS            datasets/pcam-hg/test/embeddings/y_embeddings.npy
#   K                      200
#   BANDWIDTH              optional; default uses FACTOR_DIR/summary.json
#   N_SAMPLES              1000
#   N_WARMUP               500
#   LEAPFROG_VALUES        "4 6 10 16 25"
#   PREDICTION_BATCH_SIZE  512
#   HMC_STEP_CONSTANT      0.05
#   HMC_TARGET_ACCEPT      0.8
#   SUBSAMPLE_TEST         optional
#   SWEEP_OUTPUT_PREFIX    auto -> data/sweep_leapfrog
#   SEED                   42

infer_netid() {
  case "$1" in
    /scratch/*/*)
      printf '%s\n' "$1" | awk -F/ '{print $3}'
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
  esac
}

SBATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"

NETID="${NETID:-}"
if [ -z "${NETID}" ]; then
  NETID="$(infer_netid "${PROJECT_ROOT}" 2>/dev/null || true)"
fi
if [ -z "${NETID}" ] && [ -n "${USER:-}" ]; then
  NETID="${USER}"
fi
if [ -z "${NETID}" ]; then
  echo "ERROR: Could not infer NETID. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

SCRATCH_BASE="/scratch/${NETID}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
FACTOR_DIR="${FACTOR_DIR:-datasets/pcam-hg/train/embeddings/exp2_rpchol_full}"
EMBEDDINGS="${EMBEDDINGS:-}"
TEST_EMBEDDINGS="${TEST_EMBEDDINGS:-datasets/pcam-hg/test/embeddings/projected_512.npy}"
TEST_LABELS="${TEST_LABELS:-datasets/pcam-hg/test/embeddings/y_embeddings.npy}"
K="${K:-200}"
BANDWIDTH="${BANDWIDTH:-}"
N_SAMPLES="${N_SAMPLES:-1000}"
N_WARMUP="${N_WARMUP:-500}"
LEAPFROG_VALUES="${LEAPFROG_VALUES:-4 6 10 16 25}"
PREDICTION_BATCH_SIZE="${PREDICTION_BATCH_SIZE:-512}"
HMC_STEP_CONSTANT="${HMC_STEP_CONSTANT:-0.05}"
HMC_TARGET_ACCEPT="${HMC_TARGET_ACCEPT:-0.8}"
SUBSAMPLE_TEST="${SUBSAMPLE_TEST:-}"
SEED="${SEED:-42}"

FACTOR_DIR_ABS="$(resolve_path "${FACTOR_DIR}")"
TEST_EMBEDDINGS_ABS="$(resolve_path "${TEST_EMBEDDINGS}")"
TEST_LABELS_ABS="$(resolve_path "${TEST_LABELS}")"

if [ -n "${EMBEDDINGS}" ]; then
  EMBEDDINGS_ABS="$(resolve_path "${EMBEDDINGS}")"
else
  EMBEDDINGS_ABS=""
fi

SWEEP_OUTPUT_PREFIX="${SWEEP_OUTPUT_PREFIX:-data/sweep_leapfrog}"
SWEEP_OUTPUT_PREFIX_ABS="$(resolve_path "${SWEEP_OUTPUT_PREFIX}")"
SWEEP_SUMMARY_DIR="${SWEEP_SUMMARY_DIR:-data/sweep_leapfrog_summary}"
SWEEP_SUMMARY_DIR_ABS="$(resolve_path "${SWEEP_SUMMARY_DIR}")"

if [ -f "${PROJECT_ROOT}/scripts/set_scratch_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/set_scratch_env.sh"
fi

load_conda() {
  module purge >/dev/null 2>&1 || true

  local candidates=(
    "/scratch/${NETID}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      return 0
    fi
  done

  echo "ERROR: Could not find conda.sh for NETID=${NETID}." >&2
  return 1
}

load_conda

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  exit 1
fi

unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
mkdir -p "${PROJECT_ROOT}/slurm_logs/exp2_rose"
mkdir -p "${SWEEP_SUMMARY_DIR_ABS}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, scipy, emcee; print('import check: OK')" || {
  echo "ERROR: Import check failed. Install required dependencies in ${CONDA_ENV}." >&2
  exit 1
}

cd "${PROJECT_ROOT}"

if [ ! -f "experiments/exp2_rose.py" ]; then
  echo "ERROR: experiments/exp2_rose.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${FACTOR_DIR_ABS}" ]; then
  echo "ERROR: Factor directory not found: ${FACTOR_DIR_ABS}" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/factor_k${K}.npy" ]; then
  echo "ERROR: Missing factor file: ${FACTOR_DIR_ABS}/factor_k${K}.npy" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/pivots_k${K}.npy" ]; then
  echo "ERROR: Missing pivots file: ${FACTOR_DIR_ABS}/pivots_k${K}.npy" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/kernel_submatrix_k${K}.npy" ]; then
  echo "ERROR: Missing kernel submatrix file: ${FACTOR_DIR_ABS}/kernel_submatrix_k${K}.npy" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/labels.npy" ]; then
  echo "ERROR: Missing labels file: ${FACTOR_DIR_ABS}/labels.npy" >&2
  exit 1
fi
if [ -z "${EMBEDDINGS_ABS}" ] && [ ! -f "${FACTOR_DIR_ABS}/embeddings.npy" ]; then
  echo "ERROR: No EMBEDDINGS override provided and ${FACTOR_DIR_ABS}/embeddings.npy is missing." >&2
  exit 1
fi
if [ ! -f "${TEST_EMBEDDINGS_ABS}" ]; then
  echo "ERROR: Missing test embeddings file: ${TEST_EMBEDDINGS_ABS}" >&2
  exit 1
fi
if [ ! -f "${TEST_LABELS_ABS}" ]; then
  echo "ERROR: Missing test labels file: ${TEST_LABELS_ABS}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 2 sweep: n_leapfrog grid for exp2_rose.py"
echo "=============================================================================="
echo "Job ID                 : ${SLURM_JOB_ID:-N/A}"
echo "Node                   : $(hostname)"
echo "Start time             : $(date)"
echo "Project root           : ${PROJECT_ROOT}"
echo "NETID                  : ${NETID}"
echo "Conda env              : ${CONDA_ENV}"
echo "Factor dir             : ${FACTOR_DIR}"
if [ -n "${EMBEDDINGS}" ]; then
  echo "Train embeddings       : ${EMBEDDINGS}"
else
  echo "Train embeddings       : ${FACTOR_DIR}/embeddings.npy"
fi
echo "Test embeddings        : ${TEST_EMBEDDINGS}"
echo "Test labels            : ${TEST_LABELS}"
echo "k                      : ${K}"
echo "n_samples              : ${N_SAMPLES}"
echo "n_warmup               : ${N_WARMUP}"
echo "n_leapfrog values      : ${LEAPFROG_VALUES}"
echo "Prediction batch size  : ${PREDICTION_BATCH_SIZE}"
echo "HMC step constant      : ${HMC_STEP_CONSTANT}"
echo "HMC target accept      : ${HMC_TARGET_ACCEPT}"
if [ -n "${BANDWIDTH}" ]; then
  echo "Bandwidth override     : ${BANDWIDTH}"
else
  echo "Bandwidth override     : <from ${FACTOR_DIR}/summary.json>"
fi
if [ -n "${SUBSAMPLE_TEST}" ]; then
  echo "Test subsample         : ${SUBSAMPLE_TEST}"
else
  echo "Test subsample         : <full test split>"
fi
echo "Sweep output prefix    : ${SWEEP_OUTPUT_PREFIX}_lf<value>"
echo "Sweep summary dir      : ${SWEEP_SUMMARY_DIR}"
echo "=============================================================================="

RESULTS_TSV="${SWEEP_SUMMARY_DIR_ABS}/leapfrog_results.tsv"
SUMMARY_JSON="${SWEEP_SUMMARY_DIR_ABS}/leapfrog_summary.json"

printf "n_leapfrog\ttau_nu\ttau_logp\taccept_rate\tfinal_step_size\ttau_eff_samples\ttau_logp_eff_samples\thmc_total_sec\tess_per_sec_nu\tess_per_sec_logp\tau_ok\ttau_logp_ok\tau_eval_comment\ttau_logp_eval_comment\tau_auroc\ttau_auprc\tece\taccuracy\tbrier\tn_train\tn_test\tnpz_path\n" > "${RESULTS_TSV}"

run_idx=0
for n_leapfrog in ${LEAPFROG_VALUES}; do
  run_idx=$((run_idx + 1))
  run_dir="${SWEEP_OUTPUT_PREFIX_ABS}_lf${n_leapfrog}"
  mkdir -p "${run_dir}"

  echo
  echo "[${run_idx}] Running exp2_rose.py with n_leapfrog=${n_leapfrog}"

  CMD=(
    "${PYTHON}" "experiments/exp2_rose.py"
    "--factor-dir" "${FACTOR_DIR}"
    "--test-embeddings" "${TEST_EMBEDDINGS}"
    "--test-labels" "${TEST_LABELS}"
    "--k" "${K}"
    "--n-samples" "${N_SAMPLES}"
    "--n-warmup" "${N_WARMUP}"
    "--n-leapfrog" "${n_leapfrog}"
    "--prediction-batch-size" "${PREDICTION_BATCH_SIZE}"
    "--hmc-step-constant" "${HMC_STEP_CONSTANT}"
    "--hmc-target-accept" "${HMC_TARGET_ACCEPT}"
    "--output-dir" "${run_dir}"
    "--seed" "${SEED}"
  )

  if [ -n "${EMBEDDINGS}" ]; then
    CMD+=("--embeddings" "${EMBEDDINGS}")
  fi
  if [ -n "${BANDWIDTH}" ]; then
    CMD+=("--bandwidth" "${BANDWIDTH}")
  fi
  if [ -n "${SUBSAMPLE_TEST}" ]; then
    CMD+=("--subsample-test" "${SUBSAMPLE_TEST}")
  fi

  "${CMD[@]}"

  NPZ_PATH="${run_dir}/exp2_k${K}_results.npz"
  if [ ! -f "${NPZ_PATH}" ]; then
    echo "ERROR: Missing expected result file ${NPZ_PATH}" >&2
    exit 1
  fi

  "${PYTHON}" - <<'PY' "${NPZ_PATH}" "${n_leapfrog}" "${RESULTS_TSV}"
import math
import numpy as np
import sys
from pathlib import Path

npz_path = Path(sys.argv[1])
n_leapfrog = int(sys.argv[2])
tsv_path = Path(sys.argv[3])
arr = np.load(npz_path, allow_pickle=True)

def f(name, default=float("nan")):
    if name not in arr:
        return default
    value = arr[name]
    if np.ndim(value) == 0:
        return float(value)
    return float(np.asarray(value).reshape(-1)[0])

def i(name, default=-1):
    if name not in arr:
        return default
    value = arr[name]
    if np.ndim(value) == 0:
        return int(value)
    return int(np.asarray(value).reshape(-1)[0])

tau_nu = f("tau_nu")
tau_logp = f("tau_logp")
accept_rate = f("accept_rate")
final_step_size = f("final_step_size")
auroc = f("auroc")
auprc = f("auprc")
ece = f("ece")
accuracy = f("accuracy")
brier = f("brier")
hmc_total_sec = f("hmc_total_sec", f("total_mcmc_time", float("nan")))
n_train = i("n_train")
n_test = i("n_test")
n_draws = 0
if "nu_samples" in arr:
    n_draws = int(np.asarray(arr["nu_samples"]).shape[0])

def safe_eff_samples(tau, draws):
    if math.isnan(tau) or tau <= 0 or draws <= 0:
        return float("nan")
    return draws / tau

def safe_ess_per_sec(tau, draws, total_sec):
    eff = safe_eff_samples(tau, draws)
    if math.isnan(eff) or math.isnan(total_sec) or total_sec <= 0:
        return float("nan")
    return eff / total_sec

tau_eff = safe_eff_samples(tau_nu, n_draws)
tau_logp_eff = safe_eff_samples(tau_logp, n_draws)
ess_per_sec_nu = safe_ess_per_sec(tau_nu, n_draws, hmc_total_sec)
ess_per_sec_logp = safe_ess_per_sec(tau_logp, n_draws, hmc_total_sec)

tau_ok = int(not math.isnan(tau_nu) and tau_nu > 0)
tau_logp_ok = int(not math.isnan(tau_logp) and tau_logp > 0)
tau_comment = "ok" if tau_ok else "nan_or_nonpositive"
tau_logp_comment = "ok" if tau_logp_ok else "nan_or_nonpositive"

cols = [
    str(n_leapfrog),
    f"{tau_nu:.10g}",
    f"{tau_logp:.10g}",
    f"{accept_rate:.10g}",
    f"{final_step_size:.10g}",
    f"{tau_eff:.10g}",
    f"{tau_logp_eff:.10g}",
    f"{hmc_total_sec:.10g}",
    f"{ess_per_sec_nu:.10g}",
    f"{ess_per_sec_logp:.10g}",
    str(tau_ok),
    str(tau_logp_ok),
    tau_comment,
    tau_logp_comment,
    f"{auroc:.10g}",
    f"{auprc:.10g}",
    f"{ece:.10g}",
    f"{accuracy:.10g}",
    f"{brier:.10g}",
    str(n_train),
    str(n_test),
    str(npz_path),
]

with tsv_path.open("a") as f:
    f.write("\t".join(cols) + "\n")
PY
done

"${PYTHON}" - <<'PY' "${RESULTS_TSV}" "${SUMMARY_JSON}" "${FACTOR_DIR}" "${K}" "${N_SAMPLES}" "${N_WARMUP}" "${LEAPFROG_VALUES}"
import csv
import json
import math
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
factor_dir = sys.argv[3]
k = int(sys.argv[4])
n_samples = int(sys.argv[5])
n_warmup = int(sys.argv[6])
leapfrog_values = [int(x) for x in sys.argv[7].split()]

rows = []
with tsv_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        clean = {}
        for key, value in row.items():
            if key in {"n_leapfrog", "tau_ok", "tau_logp_ok", "n_train", "n_test"}:
                clean[key] = int(value)
            elif key in {"tau_eval_comment", "tau_logp_eval_comment", "npz_path"}:
                clean[key] = value
            else:
                clean[key] = float(value)
        rows.append(clean)

if not rows:
    raise SystemExit("No sweep rows found.")

def better_key(row):
    auroc = row["tau_auroc"]
    if math.isnan(auroc):
        auroc = -float("inf")
    auprc = row["tau_auprc"]
    if math.isnan(auprc):
        auprc = -float("inf")
    ece = row["ece"]
    if math.isnan(ece):
        ece = float("inf")
    accept = row["accept_rate"]
    if math.isnan(accept):
        accept = -float("inf")
    return (auroc, auprc, -ece, accept)

best = max(rows, key=better_key)

summary = {
    "factor_dir": factor_dir,
    "k": k,
    "n_samples": n_samples,
    "n_warmup": n_warmup,
    "leapfrog_values": leapfrog_values,
    "selection_rule": "best by AUROC, then AUPRC, then lower ECE, then higher accept_rate",
    "best_n_leapfrog": best["n_leapfrog"],
    "best_metrics": {
        "auroc": best["tau_auroc"],
        "auprc": best["tau_auprc"],
        "ece": best["ece"],
        "accuracy": best["accuracy"],
        "brier": best["brier"],
        "accept_rate": best["accept_rate"],
        "tau_nu": best["tau_nu"],
        "tau_logp": best["tau_logp"],
        "tau_eff_samples": best["tau_eff_samples"],
        "tau_logp_eff_samples": best["tau_logp_eff_samples"],
        "ess_per_sec_nu": best["ess_per_sec_nu"],
        "ess_per_sec_logp": best["ess_per_sec_logp"],
        "hmc_total_sec": best["hmc_total_sec"],
    },
    "runs": rows,
}

with summary_path.open("w") as f:
    json.dump(summary, f, indent=2)

print("\nSweep summary")
print("-------------")
for row in rows:
    print(
        f"n_leapfrog={row['n_leapfrog']:>2d}  "
        f"auroc={row['tau_auroc']:.4f}  auprc={row['tau_auprc']:.4f}  "
        f"ece={row['ece']:.4f}  accept={row['accept_rate']:.4f}  "
        f"tau_nu={row['tau_nu']:.2f}  ess/sec={row['ess_per_sec_nu']:.4f}"
    )
print(
    f"\nBest n_leapfrog={best['n_leapfrog']} "
    f"(AUROC={best['tau_auroc']:.4f}, AUPRC={best['tau_auprc']:.4f}, ECE={best['ece']:.4f})"
)
print(f"Saved summary: {summary_path}")
PY

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts:"
echo "  - ${SWEEP_OUTPUT_PREFIX_ABS}_lf*/exp2_k${K}_results.npz"
echo "  - ${RESULTS_TSV}"
echo "  - ${SUMMARY_JSON}"
echo "=============================================================================="
