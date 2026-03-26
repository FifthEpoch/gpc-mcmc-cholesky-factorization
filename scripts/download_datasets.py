#!/usr/bin/env python3
"""
Dataset download helper for later experiments.

Supports:
1) PatchCamelyon (PCam) via official Google Drive file IDs
2) CAMELYON17-WILDS via the WILDS Python package
3) EMBED setup guidance (access request + optional AWS sync command)
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Dict

from tqdm import tqdm


PCAM_FILES: Dict[str, Dict[str, str]] = {
    "camelyonpatch_level_2_split_train_x.h5.gz": {
        "id": "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",
        "md5": "1571f514728f59376b705fc836ff4b63",
    },
    "camelyonpatch_level_2_split_train_y.h5.gz": {
        "id": "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
        "md5": "35c2d7259d906cfc8143347bb8e05be7",
    },
    "camelyonpatch_level_2_split_valid_x.h5.gz": {
        "id": "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
        "md5": "d8c2d60d490dbd479f8199bdfa0cf6ec",
    },
    "camelyonpatch_level_2_split_valid_y.h5.gz": {
        "id": "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
        "md5": "60a7035772fbdb7f34eb86d4420cf66a",
    },
    "camelyonpatch_level_2_split_test_x.h5.gz": {
        "id": "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
        "md5": "d5b63470df7cfa627aeec8b9dc0c066e",
    },
    "camelyonpatch_level_2_split_test_y.h5.gz": {
        "id": "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
        "md5": "2b85f58b927af9964a4c15b8f7e8f179",
    },
    "camelyonpatch_level_2_split_train_meta.csv": {
        "id": "1XoaGG3ek26YLFvGzmkKeOz54INW0fruR",
        "md5": "5a3dd671e465cfd74b5b822125e65b0a",
    },
    "camelyonpatch_level_2_split_valid_meta.csv": {
        "id": "16hJfGFCZEcvR3lr38v3XCaD5iH1Bnclg",
        "md5": "3455fd69135b66734e1008f3af684566",
    },
    "camelyonpatch_level_2_split_test_meta.csv": {
        "id": "19tj7fBlQQrd4DapCjhZrom_fA4QlHqN4",
        "md5": "67589e00a4a37ec317f2d1932c7502ca",
    },
}


def md5sum(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pcam(target_dir: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed. Install dependencies first "
            "(pip install -r requirements.txt)."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PCam] Download directory: {target_dir}")

    for filename, info in tqdm(PCAM_FILES.items(), desc="PCam files"):
        output_path = target_dir / filename
        expected_md5 = info["md5"]
        file_id = info["id"]

        if output_path.exists():
            actual_md5 = md5sum(output_path)
            if actual_md5 == expected_md5:
                print(f"[PCam] OK (already downloaded): {filename}")
                continue
            print(f"[PCam] MD5 mismatch, re-downloading: {filename}")
            output_path.unlink()

        gdown.download(id=file_id, output=str(output_path), quiet=False)
        actual_md5 = md5sum(output_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"[PCam] Checksum mismatch for {filename}. "
                f"Expected {expected_md5}, got {actual_md5}."
            )
        print(f"[PCam] Downloaded + verified: {filename}")


def download_camelyon17_wilds(target_dir: Path) -> None:
    try:
        from wilds import get_dataset
    except ImportError as exc:
        raise RuntimeError(
            "wilds is not installed. Run:\n"
            "  pip install wilds torch torchvision"
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
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
