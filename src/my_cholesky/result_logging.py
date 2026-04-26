"""Utilities for appending experiment results to a shared CSV file."""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = PROJECT_ROOT / "scripts" / "experiment_results_runs_template.csv"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "data" / "experiment_results_runs.csv"


def load_result_schema(template_path: Path = TEMPLATE_PATH) -> list[str]:
    """Load the canonical experiment-results CSV header."""
    with template_path.open(newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def default_results_csv() -> Path:
    """Return the output CSV path, allowing cluster jobs to override it."""
    return Path(os.environ.get("EXPERIMENT_RESULTS_CSV", DEFAULT_RESULTS_CSV))


def git_commit() -> str:
    """Best-effort current git commit SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def base_run_metadata() -> dict[str, Any]:
    """Common metadata available in local and Slurm environments."""
    return {
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "account_partition": "/".join(
            part
            for part in [
                os.environ.get("SLURM_JOB_ACCOUNT", ""),
                os.environ.get("SLURM_JOB_PARTITION", ""),
            ]
            if part
        ),
        "hostname": socket.gethostname(),
        "node_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "code_ref": git_commit(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conda_env": os.environ.get("CONDA_PREFIX", ""),
    }


def make_record_id(experiment: str, method_name: str = "", suffix: str = "") -> str:
    """Create a readable fallback record id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job = os.environ.get("SLURM_JOB_ID", "local")
    parts = [stamp, job, experiment, method_name, suffix]
    return "-".join(str(part) for part in parts if part)


def _serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def append_result_row(row: dict[str, Any], csv_path: str | Path | None = None) -> Path:
    """Append one row to the shared results CSV using the canonical schema."""
    output_path = Path(csv_path) if csv_path is not None else default_results_csv()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = load_result_schema()

    merged = {**base_run_metadata(), **row}
    if not merged.get("record_id"):
        merged["record_id"] = make_record_id(
            str(merged.get("experiment", "")),
            str(merged.get("method_name", "") or merged.get("sampler", "")),
            str(merged.get("dataset", "") or merged.get("k", "")),
        )

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: _serialize(merged.get(key, "")) for key in schema})

    return output_path


def append_result_rows(
    rows: Iterable[dict[str, Any]], csv_path: str | Path | None = None
) -> Path:
    """Append multiple rows and return the output CSV path."""
    output_path = Path(csv_path) if csv_path is not None else default_results_csv()
    for row in rows:
        append_result_row(row, output_path)
    return output_path
