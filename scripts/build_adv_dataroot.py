#!/usr/bin/env python3
"""
Build an evaluation dataroot for adversarial images:
  out_dataroot/real/*  -> symlinked from clean dataroot
  out_dataroot/fake/*  -> adversarial images mapped back to the original fake paths

Assumes attack output from `main.py` saved files like:
  attack_out/0000_adv_image.png
  attack_out/0001_adv_image.png
in the same order as `natsorted(clean_dataroot/fake/**)` would produce.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from natsort import natsorted, ns


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


def iter_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dataroot", required=True, help="Folder with real/ and fake/ used for attack ordering")
    ap.add_argument("--attack_out", required=True, help="Folder containing 0000_adv_image.png ...")
    ap.add_argument("--out_dataroot", required=True, help="Output dataroot (will be created)")
    ap.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    args = ap.parse_args()

    clean = Path(args.clean_dataroot)
    attack_out = Path(args.attack_out)
    out = Path(args.out_dataroot)
    out_fake = out / "fake"
    out_real = out / "real"
    ensure_dir(out_fake)
    ensure_dir(out_real)

    # Link/copy real images unchanged.
    real_src = clean / "real"
    real_files = [p for p in iter_images(real_src)]
    for p in real_files:
        dst = out_real / p.name
        link_or_copy(p, dst, args.mode)

    # Determine fake ordering and map indices to adv images.
    fake_src = clean / "fake"
    fake_files = [str(p) for p in iter_images(fake_src)]
    fake_files = natsorted(fake_files, alg=ns.PATH)

    manifest = out / "manifest_fake_adv.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "clean_path", "adv_path", "out_path"])
        for i, clean_path in enumerate(fake_files):
            adv_img = attack_out / f"{i:04d}_adv_image.png"
            if not adv_img.exists():
                raise FileNotFoundError(f"Missing {adv_img} for index {i}")

            clean_p = Path(clean_path)
            rel = clean_p.relative_to(fake_src)
            out_p = out_fake / rel
            link_or_copy(adv_img, out_p, args.mode)
            w.writerow([i, clean_path, str(adv_img), str(out_p)])

    print(f"Built adv dataroot: {out}")
    print(f"  real: {len(real_files)} images (copied/linked)")
    print(f"  fake_adv: {len(fake_files)} images (mapped from {attack_out})")
    print(f"  manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

