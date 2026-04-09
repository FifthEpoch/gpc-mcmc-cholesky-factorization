#!/usr/bin/env python3
"""
Dataset download helper for later experiments.

Supports:
1) PatchCamelyon (PCam) via Zenodo (primary) or Google Drive (fallback)
2) CAMELYON17-WILDS via the WILDS Python package
3) EMBED setup guidance (access request + optional AWS sync command)
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict


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

_ensure_package("tqdm")
from tqdm import tqdm


ZENODO_RECORD = "2546921"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

PCAM_FILES: Dict[str, Dict[str, str]] = {
    "camelyonpatch_level_2_split_train_x.h5.gz": {
        "gdrive_id": "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",
        "md5": "1571f514728f59376b705fc836ff4b63",
    },
    "camelyonpatch_level_2_split_train_y.h5.gz": {
        "gdrive_id": "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
        "md5": "35c2d7259d906cfc8143347bb8e05be7",
    },
    "camelyonpatch_level_2_split_valid_x.h5.gz": {
        "gdrive_id": "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
        "md5": "d5b63470df7cfa627aeec8b9dc0c066e",
    },
    "camelyonpatch_level_2_split_valid_y.h5.gz": {
        "gdrive_id": "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
        "md5": "2b85f58b927af9964a4c15b8f7e8f179",
    },
    "camelyonpatch_level_2_split_test_x.h5.gz": {
        "gdrive_id": "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
        "md5": "d8c2d60d490dbd479f8199bdfa0cf6ec",
    },
    "camelyonpatch_level_2_split_test_y.h5.gz": {
        "gdrive_id": "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
        "md5": "60a7035772fbdb7f34eb86d4420cf66a",
    },
    "camelyonpatch_level_2_split_train_meta.csv": {
        "gdrive_id": "1XoaGG3ek26YLFvGzmkKeOz54INW0fruR",
        "md5": "5a3dd671e465cfd74b5b822125e65b0a",
    },
    "camelyonpatch_level_2_split_valid_meta.csv": {
        "gdrive_id": "16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg",
        "md5": "67589e00a4a37ec317f2d1932c7502ca",
    },
    "camelyonpatch_level_2_split_test_meta.csv": {
        "gdrive_id": "19tj7fBlQQrd4DapCjhZrom_fA4QlHqN4",
        "md5": "3455fd69135b66734e1008f3af684566",
    },
}


def md5sum(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "pcam-downloader/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            miniters=1,
        ) as bar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def download_pcam(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PCam] Download directory: {target_dir}")
    print(f"[PCam] Using Zenodo mirror (record {ZENODO_RECORD})")

    for filename, info in PCAM_FILES.items():
        output_path = target_dir / filename
        expected_md5 = info["md5"]

        if output_path.exists():
            actual_md5 = md5sum(output_path)
            if actual_md5 == expected_md5:
                print(f"[PCam] OK (already downloaded): {filename}")
                continue
            print(f"[PCam] MD5 mismatch, re-downloading: {filename}")
            output_path.unlink()

        zenodo_url = f"{ZENODO_BASE}/{filename}?download=1"
        try:
            print(f"[PCam] Downloading from Zenodo: {filename}")
            _download_url(zenodo_url, output_path)
        except Exception as exc:
            print(f"[PCam] Zenodo download failed ({exc}), trying Google Drive ...")
            _ensure_package("gdown")
            import gdown
            gdown.download(id=info["gdrive_id"], output=str(output_path), quiet=False)

        actual_md5 = md5sum(output_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"[PCam] Checksum mismatch for {filename}. "
                f"Expected {expected_md5}, got {actual_md5}."
            )
        print(f"[PCam] Verified: {filename}")


def _fix_ssl_certs() -> None:
    """Ensure Python can verify SSL certificates (macOS framework Python
    ships without root certs; installing *certifi* and pointing
    ``SSL_CERT_FILE`` at its bundle fixes the issue)."""
    import os
    import ssl

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = True
        urllib.request.urlopen(
            urllib.request.Request("https://wilds.stanford.edu", method="HEAD"),
            context=ctx,
            timeout=5,
        )
        return  # certs already work
    except Exception:
        pass

    _ensure_package("certifi")
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    print(f"[setup] SSL_CERT_FILE set to {certifi.where()}")


def download_camelyon17_wilds(target_dir: Path) -> None:
    _fix_ssl_certs()
    _ensure_package("wilds")
    from wilds import get_dataset

    target_dir.mkdir(parents=True, exist_ok=True)

    # Clean up corrupted archive from a previous failed attempt
    archive = target_dir / "camelyon17_v1.0" / "archive.tar.gz"
    if archive.exists() and archive.stat().st_size == 0:
        print("[CAMELYON17-WILDS] Removing empty/corrupted archive from prior run ...")
        archive.unlink()

    print(f"[CAMELYON17-WILDS] Root directory: {target_dir}")
    _ = get_dataset(dataset="camelyon17", download=True, root_dir=str(target_dir))
    print("[CAMELYON17-WILDS] Download complete.")


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
