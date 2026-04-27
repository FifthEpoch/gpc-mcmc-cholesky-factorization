#!/usr/bin/env python3
"""
check_images.py
---------------
For every row in train/valid/test labels.csv, check whether the image file
actually exists on disk and print a log line. Print a summary at the end.

Usage:
    python check_images.py --root datasets/camelyon17-hg
    python check_images.py --root datasets/camelyon17-hg --image-col image_path
    python check_images.py --root datasets/camelyon17-hg --image-ext .png
    python check_images.py --root datasets/camelyon17-hg --quiet  # only misses
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path


# Common column names used for the image reference in labels.csv
IMG_COL_CANDIDATES = [
    "image", "image_path", "img", "img_path",
    "filename", "file", "path", "patch", "patch_path",
]


def detect_image_column(fieldnames):
    lower = {f.lower(): f for f in fieldnames}
    for c in IMG_COL_CANDIDATES:
        if c in lower:
            return lower[c]
    return fieldnames[0]  # fallback to first column


def resolve_image(split_dir: Path, ref: str, image_ext: str) -> Path | None:
    """Return the resolved Path if the image exists, else None."""
    if not ref:
        return None
    p = split_dir / ref
    if p.is_file():
        return p
    if image_ext and not ref.endswith(image_ext):
        p2 = split_dir / (ref + image_ext)
        if p2.is_file():
            return p2
    return None


def check_split(split: str, root: Path, image_col: str | None,
                image_ext: str, quiet: bool) -> dict:
    split_dir = root / split
    csv_path = split_dir / "labels.csv"

    if not csv_path.is_file():
        print(f"[{split}] ERROR: {csv_path} not found", file=sys.stderr)
        return {"split": split, "total": 0, "present": 0, "missing": 0}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"[{split}] ERROR: empty CSV", file=sys.stderr)
            return {"split": split, "total": 0, "present": 0, "missing": 0}
        col = image_col or detect_image_column(reader.fieldnames)
        rows = list(reader)

    print(f"\n[{split}] {csv_path}  (image column: {col!r}, {len(rows)} rows)")

    present = missing = 0
    for i, row in enumerate(rows, 1):
        ref = (row.get(col) or "").strip()
        img = resolve_image(split_dir, ref, image_ext)
        if img is not None:
            present += 1
            if not quiet:
                print(f"  PRESENT  [{split}] row={i:<7} {ref}")
        else:
            missing += 1
            print(f"  MISSING  [{split}] row={i:<7} {ref}")

    return {"split": split, "total": len(rows),
            "present": present, "missing": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="dataset root containing train/ valid/ test/")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--image-col", default=None,
                    help="CSV column with the image path (auto-detected if omitted)")
    ap.add_argument("--image-ext", default=".png",
                    help="extension to append if CSV refs are bare filenames "
                         "(default .png; pass empty string to disable)")
    ap.add_argument("--quiet", action="store_true",
                    help="only log MISSING rows, not PRESENT ones")
    args = ap.parse_args()

    summaries = [check_split(s, args.root, args.image_col,
                             args.image_ext, args.quiet)
                 for s in args.splits]

    print("\n========== SUMMARY ==========")
    print(f"{'split':<8} {'total':>10} {'present':>10} {'missing':>10}  {'%present':>8}")
    print("-" * 52)
    for s in summaries:
        pct = (100 * s['present'] / s['total']) if s['total'] else 0.0
        print(f"{s['split']:<8} {s['total']:>10} {s['present']:>10} "
              f"{s['missing']:>10}  {pct:>7.2f}%")


if __name__ == "__main__":
    main()