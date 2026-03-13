#!/usr/bin/env python3
"""
Build a *full* evaluation dataroot for adversarial images, matching the clean dataset structure:

  out_dataroot/
    real/*                 (label=1, copied/linked from clean_dataroot/real)
    fake/<subset>/*        (label=0, adversarial images mapped back to clean relative paths)
    manifest_real.csv      (src,dst)
    manifest_fake.csv      (src,dst)              # src is adv image path
    manifest_fake_adv.csv  (idx,clean_path,adv_path,out_path)
    meta.txt

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
from datetime import datetime, timezone
from pathlib import Path

import re


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
    files.sort(key=lambda q: q.as_posix())
    return files


_re_digits = re.compile(r"(\\d+)")


def _natural_key(s: str) -> list[object]:
    # Split into digit and non-digit chunks: "img12.png" -> ["img", 12, ".png"]
    out: list[object] = []
    for part in _re_digits.split(s):
        if not part:
            continue
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out


def natsorted_paths(ps: list[str]) -> list[str]:
    # PATH-like natural sort: compare by path components, each component naturally.
    def key(p: str) -> list[object]:
        parts: list[object] = []
        for comp in Path(p).as_posix().split("/"):
            parts.extend(_natural_key(comp))
            parts.append("/")  # separator marker to avoid accidental merges
        return parts

    return sorted(ps, key=key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dataroot", required=True, help="Folder with real/ and fake/ used for attack ordering")
    ap.add_argument("--attack_out", required=True, help="Folder containing 0000_adv_image.png ...")
    ap.add_argument("--out_dataroot", required=True, help="Output dataroot (must not exist)")
    ap.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    ap.add_argument("--baseline", default="StealthDiffusion", help="Baseline/attack name for meta.txt")
    ap.add_argument("--surrogate", default="", help="Surrogate detector tag (E/R/D/S) for meta.txt")
    ap.add_argument(
        "--order_manifest",
        default="",
        help=(
            "Optional CSV (manifest_fake_adv.csv) to define the exact fake ordering. "
            "If provided, we map idx -> fake/<subset>/<name> using the clean_path column and ignore filesystem sorting."
        ),
    )
    args = ap.parse_args()

    clean = Path(args.clean_dataroot)
    attack_out = Path(args.attack_out)
    out = Path(args.out_dataroot)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"out_dataroot already exists: {out}")

    out_fake = out / "fake"
    out_real = out / "real"
    ensure_dir(out_fake)
    ensure_dir(out_real)

    # Real (unchanged)
    real_src = clean / "real"
    real_files = iter_images(real_src)
    real_manifest = out / "manifest_real.csv"
    with real_manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        for p in real_files:
            # Resolve so the out dataroot does not depend on a chain of symlinks.
            src = p.resolve()
            dst = out_real / p.name
            link_or_copy(src, dst, args.mode)
            w.writerow([str(src), str(dst)])

    # Fake (adv mapped back)
    fake_src = clean / "fake"
    if args.order_manifest:
        order_csv = Path(args.order_manifest)
        if not order_csv.exists():
            raise FileNotFoundError(f"order_manifest not found: {order_csv}")
        ordered: list[tuple[int, Path]] = []
        with order_csv.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                idx = int(row["idx"])
                clean_path = Path(row["clean_path"])
                # clean_path looks like ".../fake/<subset>/<name>" (often relative to StealthDiffusion/exp).
                parts = [p.lower() for p in clean_path.parts]
                if "fake" not in parts:
                    raise ValueError(f"Can't parse clean_path from order_manifest: {clean_path}")
                i = parts.index("fake")
                rel = Path(*clean_path.parts[i + 1 :])
                ordered.append((idx, fake_src / rel))
        ordered.sort(key=lambda t: t[0])
        fake_files = [str(p) for _, p in ordered]
    else:
        fake_files = [str(p) for p in iter_images(fake_src)]
        fake_files = natsorted_paths(fake_files)

    fake_manifest = out / "manifest_fake.csv"
    fake_adv_manifest = out / "manifest_fake_adv.csv"
    with fake_manifest.open("w", newline="") as f_fake, fake_adv_manifest.open("w", newline="") as f_adv:
        w_fake = csv.writer(f_fake)
        w_adv = csv.writer(f_adv)
        w_fake.writerow(["src", "dst"])
        w_adv.writerow(["idx", "clean_path", "adv_path", "out_path"])

        for i, clean_path in enumerate(fake_files):
            adv_img = attack_out / f"{i:04d}_adv_image.png"
            if not adv_img.exists():
                raise FileNotFoundError(f"Missing {adv_img} for index {i}")

            clean_p = Path(clean_path)
            rel = clean_p.relative_to(fake_src)
            out_p = out_fake / rel

            link_or_copy(adv_img, out_p, args.mode)
            w_fake.writerow([str(adv_img), str(out_p)])
            w_adv.writerow([i, str(clean_p), str(adv_img), str(out_p)])

    meta = out / "meta.txt"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta.write_text(
        "\n".join(
            [
                f"created_utc={now}",
                f"baseline={args.baseline}",
                f"surrogate={args.surrogate}",
                f"mode={args.mode}",
                f"clean_dataroot={clean.resolve()}",
                f"attack_out={attack_out.resolve()}",
                f"n_real={len(real_files)}",
                f"n_fake={len(fake_files)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Built attack dataroot: {out}")
    print(f"  real: {len(real_files)} images")
    print(f"  fake_adv: {len(fake_files)} images (mapped from {attack_out})")
    print(f"  wrote: {real_manifest.name}, {fake_manifest.name}, {fake_adv_manifest.name}, {meta.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
