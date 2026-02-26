#!/usr/bin/env python3
"""
Prepare GenImage subset splits based on filename pattern.

Assumption (per user): for files like `003_sdv4_00066.png`, the subset name is the
token between the first two underscores (i.e., stem.split("_")[1]).

Creates:
- out_dir/fake_by_subset/<subset>/*  (all images, organized)
- out_dir/fake_100xN/<subset>/*      (sampled K per subset for top-N eligible subsets)

Links are symlinks by default to avoid duplicating data.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def infer_subset_from_name(p: Path) -> str:
    stem = p.stem
    parts = stem.split("_")
    if len(parts) < 2:
        return "unknown"
    # Common GenImage patterns:
    # - `003_sdv4_00066.png`      -> subset is `sdv4` (token[1])
    # - `1_adm_180.PNG`           -> subset is `adm`  (token[1])
    # - `GLIDE_1000_..._glide_..` -> subset is `GLIDE` (token[0])
    # - `VQDM_1000_..._vqdm_..`   -> subset is `VQDM`  (token[0])
    if parts[0].isdigit():
        return (parts[1] if len(parts) >= 2 else "unknown").lower()
    return parts[0].lower()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    if mode == "symlink":
        # Use relative symlinks so the folder can be moved as a unit.
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dir", required=True, help="Path to GenImage fake directory")
    ap.add_argument("--out_dir", required=True, help="Output folder (will be created)")
    ap.add_argument("--k", type=int, default=100, help="Images per subset to sample")
    ap.add_argument("--n_sets", type=int, default=8, help="How many subsets to include in the K-per-subset sample")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help="How to materialize files in out_dir",
    )
    args = ap.parse_args()

    fake_dir = Path(args.fake_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    files = [p for p in fake_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name)

    by_subset: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_subset[infer_subset_from_name(p)].append(p)

    counts = Counter({k: len(v) for k, v in by_subset.items()})
    # Print counts in a stable order (desc, then name)
    print("Subset counts (fake):")
    for subset, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {subset:>12s} : {cnt}")

    # Organize all images by subset.
    all_root = out_dir / "fake_by_subset"
    manifest_all = out_dir / "manifest_all.csv"
    ensure_dir(all_root)
    with manifest_all.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "src", "dst"])
        for subset, ps in sorted(by_subset.items(), key=lambda kv: kv[0]):
            subset_dir = all_root / subset
            ensure_dir(subset_dir)
            for p in ps:
                dst = subset_dir / p.name
                link_or_copy(p, dst, args.mode)
                w.writerow([subset, str(p), str(dst)])

    # Sample K per subset for up to N subsets with >=K files.
    eligible = [k for k, v in counts.items() if v >= args.k]
    eligible.sort(key=lambda s: (-counts[s], s))
    chosen = eligible[: args.n_sets]

    rng = random.Random(args.seed)
    sample_root = out_dir / f"fake_{args.k}x{len(chosen)}"
    manifest_sample = out_dir / f"manifest_fake_{args.k}x{len(chosen)}.csv"
    ensure_dir(sample_root)

    total = 0
    with manifest_sample.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "src", "dst"])
        for subset in chosen:
            ps = by_subset[subset][:]
            rng.shuffle(ps)
            take = ps[: args.k]
            subset_dir = sample_root / subset
            ensure_dir(subset_dir)
            for p in take:
                dst = subset_dir / p.name
                link_or_copy(p, dst, args.mode)
                w.writerow([subset, str(p), str(dst)])
            total += len(take)

    print()
    if len(chosen) < args.n_sets:
        print(
            f"WARNING: Only {len(chosen)} subsets have >= {args.k} images "
            f"(requested {args.n_sets})."
        )
    print(f"Prepared sample folder: {sample_root} ({total} images)")
    print(f"All-by-subset folder:   {all_root} ({len(files)} images)")
    print(f"Manifests written:      {manifest_all.name}, {manifest_sample.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
