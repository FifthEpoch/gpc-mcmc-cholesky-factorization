#!/usr/bin/env python3
"""
Dataset download helper for later experiments.

Supports:
1) PatchCamelyon (PCam)   — via HuggingFace (1aurent/PatchCamelyon)
2) CAMELYON17-WILDS       — via HuggingFace (wltjr1007/Camelyon17-WILDS)
3) EMBED                  — access-gated (guidance + optional AWS S3 sync)

Both PCam and CAMELYON17 use HuggingFace ``datasets`` for reliable,
rate-limit-free downloads (the original Google Drive / CodaLab sources
are broken or frequently rate-limited).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ensure_package(package: str, pip_name: str | None = None) -> None:
    """Import *package*; if missing, install it via pip and retry."""
    try:
        __import__(package)
    except ImportError:
        install = pip_name or package
        print(f"[setup] {package!r} not found — installing via pip ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", install]
        )


HF_PCAM_REPO = "1aurent/PatchCamelyon"
HF_CAMELYON17_REPO = "wltjr1007/Camelyon17-WILDS"


def _setup_hf_cache(cache_dir: str) -> None:
    """Point HuggingFace downloads at *cache_dir* so nothing lands in ~/.cache."""
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)


def _download_hf_dataset(repo: str, tag: str, cache_dir: str) -> None:
    """Download and verify a HuggingFace dataset, split by split."""
    from datasets import load_dataset

    ds_dict = load_dataset(repo, cache_dir=cache_dir)
    for split_name, split_ds in ds_dict.items():
        n = len(split_ds)
        cols = split_ds.column_names
        print(f"[{tag}]   {split_name}: {n:,} samples, columns: {cols}")
    print(f"[{tag}] Download complete.")


def download_pcam(target_dir: Path) -> None:
    """Download PatchCamelyon from the HuggingFace mirror."""
    _ensure_package("datasets")

    target_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = str(target_dir / "hf_cache")
    _setup_hf_cache(hf_cache)

    print(f"[PCam] Downloading from HuggingFace: {HF_PCAM_REPO}")
    print(f"[PCam] Cache directory: {hf_cache}")
    _download_hf_dataset(HF_PCAM_REPO, "PCam", hf_cache)


def download_camelyon17_wilds(target_dir: Path) -> None:
    """Download CAMELYON17-WILDS from the HuggingFace mirror.

    The original CodaLab source used by the ``wilds`` library has been
    permanently broken since June 2025 (HTTP 500).  This function uses
    the community HuggingFace mirror instead.
    """
    _ensure_package("datasets")

    target_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = str(target_dir / "hf_cache")
    _setup_hf_cache(hf_cache)

    print(f"[CAMELYON17-WILDS] Downloading from HuggingFace: {HF_CAMELYON17_REPO}")
    print(f"[CAMELYON17-WILDS] Cache directory: {hf_cache}")
    _download_hf_dataset(HF_CAMELYON17_REPO, "CAMELYON17-WILDS", hf_cache)


def handle_embed(target_dir: Path, embed_s3_uri: str | None) -> None:
    embed_dir = target_dir / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)

    print("[EMBED] Direct public download is not available without approval.")
    print("[EMBED] Access request form: https://forms.gle/6YVFKTz7ucEJKEWw8")
    print("[EMBED] Documentation: https://docs.hitilab.com/")
    print("[EMBED] Data use agreement:")
    print("        https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/EMBED_license.md")

    if embed_s3_uri:
        print(f"[EMBED] Syncing from approved S3 URI: {embed_s3_uri}")
        cmd = ["aws", "s3", "sync", embed_s3_uri, str(embed_dir), "--no-sign-request"]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "aws CLI not found. Install AWS CLI and configure credentials, then rerun."
            ) from exc
        print("[EMBED] Sync complete.")
    else:
        print("[EMBED] After approval, rerun with:")
        print(
            "        python scripts/download_datasets.py "
            "--datasets embed --embed-s3-uri s3://<approved-bucket-or-prefix>"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets for later experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["pcam", "camelyon17", "embed", "all"],
        default=["all"],
        help="Which datasets to fetch.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="Local root folder for downloaded datasets.",
    )
    parser.add_argument(
        "--embed-s3-uri",
        type=str,
        default=None,
        help="Approved EMBED S3 URI (if available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.datasets)
    if "all" in selected:
        selected = {"pcam", "camelyon17", "embed"}

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {root}")

    if "pcam" in selected:
        download_pcam(root / "pcam")
    if "camelyon17" in selected:
        download_camelyon17_wilds(root / "camelyon17_wilds")
    if "embed" in selected:
        handle_embed(root, args.embed_s3_uri)

    print("Done.")


if __name__ == "__main__":
    main()
