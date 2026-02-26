#!/usr/bin/env python3
"""
Create an evaluation dataroot with:
  out_dir/real/*  (sampled genuine images)
  out_dir/fake/*  (provided fake sample folder, optionally with subset subfolders)

Uses symlinks by default to avoid duplicating data and writes a manifest for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    if mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def iter_images(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    files: list[Path] = []
    for x in p.rglob("*"):
        if x.is_file() and x.suffix.lower() in IMG_EXTS:
            files.append(x)
    files.sort(key=lambda q: q.as_posix())
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_sample_dir", required=True, help="Folder containing selected fake images (may contain subset subfolders)")
    ap.add_argument("--real_dir", required=True, help="Folder containing genuine/real images")
    ap.add_argument("--real_k", type=int, default=700, help="How many real images to sample")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    ap.add_argument("--out_dir", required=True, help="Output dataroot")
    ap.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fake_src = Path(args.fake_sample_dir)
    real_src = Path(args.real_dir)

    out_fake = out_dir / "fake"
    out_real = out_dir / "real"
    ensure_dir(out_fake)
    ensure_dir(out_real)

    # Copy/link fake sample folder as-is under out_dir/fake.
    fake_files = iter_images(fake_src)
    fake_manifest = out_dir / "manifest_fake.csv"
    with fake_manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        for p in fake_files:
            rel = p.relative_to(fake_src)
            dst = out_fake / rel
            link_or_copy(p, dst, args.mode)
            w.writerow([str(p), str(dst)])
    print(f"fake: {len(fake_files)} images -> {out_fake}")

    # Sample real images into out_dir/real.
    real_files = [p for p in real_src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    real_files.sort(key=lambda q: q.name)
    if len(real_files) < args.real_k:
        raise ValueError(f"real_dir has only {len(real_files)} images, need real_k={args.real_k}")

    rng = random.Random(args.seed)
    rng.shuffle(real_files)
    real_take = real_files[: args.real_k]

    real_manifest = out_dir / "manifest_real.csv"
    with real_manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        for p in sorted(real_take, key=lambda q: q.name):
            dst = out_real / p.name
            link_or_copy(p, dst, args.mode)
            w.writerow([str(p), str(dst)])
    print(f"real: {len(real_take)} images -> {out_real}")

    meta = out_dir / "meta.txt"
    meta.write_text(
        f"seed={args.seed}\nmode={args.mode}\nreal_k={args.real_k}\n"
        f"fake_sample_dir={fake_src}\nreal_dir={real_src}\n",
        encoding="utf-8",
    )
    print(f"wrote: {fake_manifest.name}, {real_manifest.name}, {meta.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

