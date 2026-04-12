#!/usr/bin/env python3
"""
Dataset download helper for later experiments.

Supports:
1) PatchCamelyon (PCam) via official Google Drive file IDs, Hugging Face parquet
   shards, or existing local archives
2) Optional PCam split preparation under train/valid/test
3) CAMELYON17-WILDS via the WILDS Python package
4) EMBED setup guidance (access request + optional AWS sync command)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
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

PCAM_SPLITS: Dict[str, Dict[str, str]] = {
    "train": {
        "x_gz": "camelyonpatch_level_2_split_train_x.h5.gz",
        "y_gz": "camelyonpatch_level_2_split_train_y.h5.gz",
        "meta": "camelyonpatch_level_2_split_train_meta.csv",
    },
    "valid": {
        "x_gz": "camelyonpatch_level_2_split_valid_x.h5.gz",
        "y_gz": "camelyonpatch_level_2_split_valid_y.h5.gz",
        "meta": "camelyonpatch_level_2_split_valid_meta.csv",
    },
    "test": {
        "x_gz": "camelyonpatch_level_2_split_test_x.h5.gz",
        "y_gz": "camelyonpatch_level_2_split_test_y.h5.gz",
        "meta": "camelyonpatch_level_2_split_test_meta.csv",
    },
}


def md5sum(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pcam_google_drive(target_dir: Path) -> None:
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


def download_pcam_huggingface(target_dir: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install dependencies first "
            "(pip install -r requirements.txt)."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = target_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PCam] Downloading from Hugging Face: {repo_id}")
    print(f"[PCam] Cache directory: {cache_dir}")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["data/*.parquet", "README.md"],
            cache_dir=str(cache_dir),
        )
    )

    data_dir = snapshot_dir / "data"
    if not data_dir.exists():
        raise RuntimeError(f"[PCam] Expected Hugging Face data directory at {data_dir}")

    split_counts = {"train": 0, "valid": 0, "test": 0}
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        if parquet_path.name.startswith("train-"):
            split = "train"
        elif parquet_path.name.startswith("valid-"):
            split = "valid"
        elif parquet_path.name.startswith("test-"):
            split = "test"
        else:
            print(f"[PCam] Skipping unrecognized shard: {parquet_path.name}")
            continue

        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / parquet_path.name

        if output_path.exists() and output_path.stat().st_size == parquet_path.stat().st_size:
            print(f"[PCam] OK (already copied): {split}/{parquet_path.name}")
        else:
            shutil.copy2(parquet_path, output_path)
            print(f"[PCam] Saved {split}/{parquet_path.name}")
        split_counts[split] += 1

    print(
        "[PCam] Hugging Face download complete. "
        f"Splits: train={split_counts['train']}, "
        f"valid={split_counts['valid']}, test={split_counts['test']}"
    )


def copy_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"[PCam] OK (already exists): {dst}")
        return

    shutil.copy2(src, dst)
    print(f"[PCam] Copied: {src.name} -> {dst}")


def gunzip_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"[PCam] OK (already decompressed): {dst}")
        return

    with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    print(f"[PCam] Decompressed: {src.name} -> {dst}")


def prepare_pcam_split_layout(target_dir: Path) -> None:
    has_local_archives = False
    for split_files in PCAM_SPLITS.values():
        x_gz = target_dir / split_files["x_gz"]
        x_h5 = target_dir / split_files["x_gz"].removesuffix(".gz")
        y_gz = target_dir / split_files["y_gz"]
        y_h5 = target_dir / split_files["y_gz"].removesuffix(".gz")
        if x_gz.exists() or x_h5.exists() or y_gz.exists() or y_h5.exists():
            has_local_archives = True
            break

    if not has_local_archives:
        print(
            "[PCam] No local .h5.gz/.h5 archives found under "
            f"{target_dir}; skipping HDF5 split preparation."
        )
        return

    missing_files = []
    for split, split_files in PCAM_SPLITS.items():
        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for key, dest_name in (("x_gz", "x.h5"), ("y_gz", "y.h5")):
            gz_name = split_files[key]
            root_gz_path = target_dir / gz_name
            root_h5_path = target_dir / gz_name.removesuffix(".gz")
            dest_path = split_dir / dest_name

            if dest_path.exists():
                print(f"[PCam] OK (already prepared): {dest_path}")
            elif root_h5_path.exists():
                copy_if_needed(root_h5_path, dest_path)
            elif root_gz_path.exists():
                gunzip_if_needed(root_gz_path, dest_path)
            else:
                missing_files.append(root_gz_path.name)

        meta_src = target_dir / split_files["meta"]
        meta_dst = split_dir / "meta.csv"
        if meta_src.exists():
            copy_if_needed(meta_src, meta_dst)
        else:
            missing_files.append(meta_src.name)

    if missing_files:
        unique_missing = ", ".join(sorted(set(missing_files)))
        raise RuntimeError(
            "[PCam] Could not fully prepare split layout because the following files "
            f"were not found in {target_dir}: {unique_missing}"
        )

    print(
        "[PCam] Prepared split layout under "
        f"{target_dir} with train/, valid/, and test/ directories."
    )


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
    parser.add_argument(
        "--pcam-source",
        choices=["gdrive", "hf", "existing"],
        default="gdrive",
        help=(
            "Where to source PCam from: official Google Drive files, Hugging Face "
            "parquet shards, or existing local files already under --root/pcam."
        ),
    )
    parser.add_argument(
        "--pcam-hf-repo",
        type=str,
        default="1aurent/PatchCamelyon",
        help="Hugging Face dataset repo to use when --pcam-source hf.",
    )
    parser.add_argument(
        "--prepare-pcam",
        action="store_true",
        help=(
            "Create pcam/train, pcam/valid, and pcam/test. For local PCam archives, "
            "decompress x/y .h5.gz files into each split folder and copy meta.csv."
        ),
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
        pcam_dir = root / "pcam"
        if args.pcam_source == "gdrive":
            download_pcam_google_drive(pcam_dir)
        elif args.pcam_source == "hf":
            download_pcam_huggingface(pcam_dir, args.pcam_hf_repo)
        else:
            pcam_dir.mkdir(parents=True, exist_ok=True)
            print(f"[PCam] Reusing existing local files in: {pcam_dir}")

        if args.prepare_pcam:
            prepare_pcam_split_layout(pcam_dir)
    if "camelyon17" in selected:
        download_camelyon17_wilds(root / "camelyon17_wilds")
    if "embed" in selected:
        handle_embed(root, args.embed_s3_uri)

    print("Done.")


if __name__ == "__main__":
    main()
