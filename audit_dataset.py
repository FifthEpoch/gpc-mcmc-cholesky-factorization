#!/usr/bin/env python3
"""
audit_dataset.py
----------------
Audit a Camelyon17-style dataset to find out:
  1. Which rows in labels.csv reference an image that exists on disk.
  2. Which rows reference an image that is MISSING.
  3. Which images on disk are ORPHANS (no row in labels.csv).
  4. Whether each present image has a matching embedding file.

It writes:
  * Per-split log files (one line per image, easy to grep)
  * A summary printed to stdout
  * A `*_clean.csv` for each split with only rows whose image AND embedding exist
    (so the WARNING about skipped rows goes away)

Usage:
  python audit_dataset.py --root datasets/camelyon17-hg \
                          --embeddings datasets/camelyon17-hg/embeddings \
                          --emb-ext .npy

Adjust --image-col / --image-ext / --emb-ext if your layout differs.
"""

from __future__ import annotations
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Logging setup: console + per-split file handlers added later
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("audit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_image_column(fieldnames: Iterable[str]) -> str:
    """Best-effort guess of which CSV column holds the image filename/path."""
    candidates = [
        "image", "image_path", "img", "img_path",
        "filename", "file", "path", "patch",
    ]
    lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in lower:
            return lower[c]
    # Fallback: first column
    first = next(iter(fieldnames))
    log.warning("Could not detect image column; falling back to '%s'", first)
    return first


def find_image_on_disk(split_dir: Path, ref: str, image_ext: str | None) -> Path | None:
    """
    Resolve a CSV reference to an actual file under split_dir.

    Tries, in order:
      - ref as given (relative to split_dir)
      - ref + image_ext
      - rglob match on basename (slow, only if needed)
    """
    p = split_dir / ref
    if p.is_file():
        return p
    if image_ext:
        p2 = split_dir / (ref + image_ext)
        if p2.is_file():
            return p2
    # Last resort: search by basename. Only do this once-per-miss; cheap enough
    # for a few thousand misses, would need indexing for millions.
    base = Path(ref).name
    matches = list(split_dir.rglob(base))
    if matches:
        return matches[0]
    if image_ext and not base.endswith(image_ext):
        matches = list(split_dir.rglob(base + image_ext))
        if matches:
            return matches[0]
    return None


def embedding_path_for(image_path: Path, split_dir: Path,
                       emb_root: Path | None, emb_ext: str) -> Path:
    """
    Mirror the image's relative path under emb_root, swapping the extension.
    If emb_root is None, put embeddings next to images.
    """
    rel = image_path.relative_to(split_dir).with_suffix(emb_ext)
    base = emb_root if emb_root else split_dir
    return base / rel


# ---------------------------------------------------------------------------
# Per-split audit
# ---------------------------------------------------------------------------
def audit_split(split: str, root: Path, emb_root: Path | None,
                image_col: str | None, image_ext: str | None,
                emb_ext: str, log_dir: Path) -> dict:
    split_dir = root / split
    csv_path = split_dir / "labels.csv"
    if not csv_path.is_file():
        log.error("[%s] labels.csv not found at %s", split, csv_path)
        return {}

    # Per-split file log
    fh = logging.FileHandler(log_dir / f"{split}_audit.log", mode="w")
    fh.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    log.addHandler(fh)

    log.info("[%s] reading %s", split, csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            log.error("[%s] CSV has no header", split)
            log.removeHandler(fh); fh.close()
            return {}
        col = image_col or detect_image_column(reader.fieldnames)
        log.info("[%s] using image column: %r", split, col)
        rows = list(reader)
        fieldnames = reader.fieldnames

    n_total = len(rows)
    present_rows, missing_img_rows, missing_emb_rows = [], [], []
    seen_images: set[Path] = set()

    emb_split_root = (emb_root / split) if emb_root else None

    for i, row in enumerate(rows, 1):
        ref = (row.get(col) or "").strip()
        if not ref:
            log.warning("[%s] row %d has empty %s — skipping", split, i, col)
            missing_img_rows.append(row)
            continue

        img = find_image_on_disk(split_dir, ref, image_ext)
        if img is None:
            log.info("MISSING_IMG  [%s] row=%d ref=%s", split, i, ref)
            missing_img_rows.append(row)
            continue

        seen_images.add(img.resolve())
        emb = embedding_path_for(img, split_dir, emb_split_root, emb_ext)
        if not emb.is_file():
            log.info("MISSING_EMB  [%s] row=%d img=%s expected_emb=%s",
                     split, i, img.name, emb)
            missing_emb_rows.append(row)
            continue

        log.debug("OK           [%s] row=%d img=%s emb=%s",
                  split, i, img.name, emb.name)
        present_rows.append(row)

    # Orphan scan: images on disk that no CSV row pointed at
    on_disk = set()
    exts = {image_ext.lower()} if image_ext else {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            on_disk.add(p.resolve())
    orphans = on_disk - seen_images
    for o in sorted(orphans):
        log.info("ORPHAN_IMG   [%s] %s", split, o.relative_to(split_dir))

    # Write cleaned CSV (rows whose image AND embedding both exist)
    clean_csv = split_dir / "labels_clean.csv"
    with clean_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(present_rows)
    log.info("[%s] wrote %s with %d rows", split, clean_csv, len(present_rows))

    summary = {
        "split": split,
        "rows_total": n_total,
        "rows_ok": len(present_rows),
        "rows_missing_image": len(missing_img_rows),
        "rows_missing_embedding": len(missing_emb_rows),
        "images_on_disk": len(on_disk),
        "orphan_images": len(orphans),
    }
    log.info("[%s] summary: %s", split, summary)
    log.removeHandler(fh); fh.close()
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="dataset root containing train/ valid/ test/")
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="root of embeddings (mirrors split layout). "
                         "If omitted, embeddings are looked for next to images.")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--image-col", default=None,
                    help="CSV column name that holds the image path/filename. "
                         "Auto-detected if omitted.")
    ap.add_argument("--image-ext", default=".png",
                    help="image extension to append if CSV refs are bare names "
                         "(set to '' to disable)")
    ap.add_argument("--emb-ext", default=".npy",
                    help="embedding file extension (.npy, .pt, .npz, ...)")
    ap.add_argument("--log-dir", type=Path, default=Path("audit_logs"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    image_ext = args.image_ext or None

    summaries = []
    for split in args.splits:
        s = audit_split(split, args.root, args.embeddings,
                        args.image_col, image_ext, args.emb_ext, args.log_dir)
        if s:
            summaries.append(s)

    # Pretty final table
    print("\n=========== AUDIT SUMMARY ===========")
    cols = ["split", "rows_total", "rows_ok",
            "rows_missing_image", "rows_missing_embedding",
            "images_on_disk", "orphan_images"]
    widths = {c: max(len(c), max((len(str(s[c])) for s in summaries), default=0))
              for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for s in summaries:
        print(" | ".join(str(s[c]).ljust(widths[c]) for c in cols))
    print(f"\nPer-split logs: {args.log_dir.resolve()}")
    print("Cleaned CSVs:   <split>/labels_clean.csv")


if __name__ == "__main__":
    main()
