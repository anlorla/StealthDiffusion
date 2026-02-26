#!/usr/bin/env python3
"""
Run baseline attacks / transformations on a folder of fake images and save outputs in the
same index-based format used by `diff_latent_attack.py`:

  0000_originImage.png
  0000_adv_image.png
  0001_originImage.png
  0001_adv_image.png
  ...

This makes it compatible with:
  - scripts/build_adv_dataroot.py (map adv back into fake/<subset>/...)
  - scripts/eval_detectors.py (compute metrics + export per-image logits CSV)

Baselines (inspired by StealthDiffusion_BigGAN/*):
  - downup: downsample then upsample (no gradients)
  - fgsm:  FGSM on surrogate detector
  - pgd:   PGD on surrogate detector
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchattacks
from natsort import natsorted, ns
from PIL import Image
from torchvision import transforms

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from loggers import Logger
import other_attacks


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def iter_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files = natsorted([str(p) for p in files], alg=ns.PATH)
    return [Path(p) for p in files]


def infer_subset_from_path(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    for i, x in enumerate(parts):
        if x == "fake" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if Path(nxt).suffix == "":
                return nxt
    return "unknown"


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    # img1,img2: uint8 HxWxC
    mse = float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # Same implementation style as diff_latent_attack.py
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def pil_downup(
    img: Image.Image,
    scale_w: float,
    scale_h: float,
    down_filter: str,
    up_filter: str,
) -> Image.Image:
    w, h = img.size
    dw, dh = max(1, int(w * scale_w)), max(1, int(h * scale_h))
    down = img.resize((dw, dh), getattr(Image, down_filter))
    up = down.resize((w, h), getattr(Image, up_filter))
    return up


def logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if logits.shape[1] == 2:
        return logits[:, 1]
    if logits.shape[1] == 1:
        return logits[:, 0]
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


@torch.no_grad()
def predict_label(model: torch.nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
    """
    Return predicted label (0=fake, 1=real) given normalized input.
    """
    logits = model(x_norm)
    score = logits_to_score(logits)
    return (score > 0).long()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", choices=["downup", "fgsm", "pgd"], required=True)
    ap.add_argument("--images_root", required=True, help="Folder containing fake images (can have subset subfolders)")
    ap.add_argument("--save_dir", required=True, help="Output folder")
    ap.add_argument("--model_name", default="S,E,R,D", help="Comma-separated detector order; first is surrogate")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_images", type=int, default=0, help=">0: only process first N images")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for fgsm/pgd")

    # fgsm/pgd
    ap.add_argument("--eps", type=float, default=4 / 255)
    ap.add_argument("--pgd_alpha", type=float, default=1 / 255)
    ap.add_argument("--pgd_steps", type=int, default=10)

    # downup
    ap.add_argument("--scale_w", type=float, default=0.5)
    ap.add_argument("--scale_h", type=float, default=0.75)
    ap.add_argument("--down_filter", type=str, default="LANCZOS")
    ap.add_argument("--up_filter", type=str, default="BICUBIC")

    args = ap.parse_args()
    seed_all(args.seed)

    images_root = Path(args.images_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(name=f"baseline_{args.attack}", log_path=str(save_dir / f"{args.attack}.log"))

    all_images = iter_images(images_root)
    if args.num_images and args.num_images > 0:
        all_images = all_images[: args.num_images]
    if not all_images:
        raise FileNotFoundError(f"No image files found under {images_root}")

    # We run baselines on fake images only (label=0) by design.
    y = torch.zeros((len(all_images),), dtype=torch.long)

    # Device
    try:
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"device={device}")

    # Load detectors
    names = [x.strip() for x in args.model_name.split(",") if x.strip()]
    if len(names) != 4:
        raise ValueError("--model_name must have 4 comma-separated names, e.g. S,E,R,D")
    logger.info(f"detectors={names}")
    detectors = [other_attacks.model_selection(n, device=device).eval() for n in names]

    tfm = transforms.Compose(
        [
            transforms.Resize((args.res, args.res), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Prepare attack object if needed
    attack = None
    if args.attack == "fgsm":
        attack = torchattacks.FGSM(detectors[0], eps=float(args.eps))
        attack.set_normalization_used(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        logger.info(f"FGSM eps={args.eps}")
    elif args.attack == "pgd":
        attack = torchattacks.PGD(
            detectors[0],
            eps=float(args.eps),
            alpha=float(args.pgd_alpha),
            steps=int(args.pgd_steps),
            random_start=False,
        )
        attack.set_normalization_used(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        logger.info(f"PGD eps={args.eps} alpha={args.pgd_alpha} steps={args.pgd_steps}")
    else:
        logger.info(
            f"DownUp scale_w={args.scale_w} scale_h={args.scale_h} "
            f"down_filter={args.down_filter} up_filter={args.up_filter}"
        )

    # Stats accumulators
    psnrs: List[float] = []
    ssims: List[float] = []
    clean_correct = [0, 0, 0, 0]
    adv_correct = [0, 0, 0, 0]

    manifest = save_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "src_path", "subset", "origin_path", "adv_path"])

        # Batch loop for speed on fgsm/pgd; downup stays per-image.
        if args.attack in ("fgsm", "pgd"):
            for start in range(0, len(all_images), args.batch):
                batch_paths = all_images[start : start + args.batch]

                imgs_pil = [Image.open(p).convert("RGB") for p in batch_paths]
                x = torch.stack([tfm(im) for im in imgs_pil], dim=0).to(device)
                yb = torch.zeros((x.shape[0],), dtype=torch.long, device=device)  # fake label=0

                # Clean predictions
                x_norm = normalize(x)
                with torch.no_grad():
                    for di, det in enumerate(detectors):
                        pred = predict_label(det, x_norm)
                        clean_correct[di] += int((pred == 0).sum().item())

                # Craft adv
                assert attack is not None
                x_adv = attack(x, yb)
                x_adv = torch.clamp(x_adv, 0, 1)

                # Adv predictions
                x_adv_norm = normalize(x_adv)
                with torch.no_grad():
                    for di, det in enumerate(detectors):
                        pred = predict_label(det, x_adv_norm)
                        adv_correct[di] += int((pred == 0).sum().item())

                # Save outputs + metrics
                for j, (p, im_pil) in enumerate(zip(batch_paths, imgs_pil)):
                    idx = start + j
                    subset = infer_subset_from_path(p)
                    origin_path = save_dir / f"{idx:04d}_originImage.png"
                    adv_path = save_dir / f"{idx:04d}_adv_image.png"

                    # Origin (resized)
                    im_resized = im_pil.resize((args.res, args.res), resample=Image.BICUBIC)
                    im_resized.save(origin_path)

                    # Adv
                    adv_np = (
                        x_adv[j].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
                    ).clip(0, 255).astype(np.uint8)
                    adv_pil = Image.fromarray(adv_np)
                    adv_pil.save(adv_path)

                    # Metrics
                    a = np.array(im_resized).astype(np.uint8)
                    psnrs.append(calculate_psnr(a, adv_np))
                    try:
                        ssims.append(calculate_ssim(a, adv_np))
                    except Exception:
                        pass

                    w.writerow([idx, str(p), subset, str(origin_path), str(adv_path)])

                logger.info(f"processed {min(start+args.batch, len(all_images))}/{len(all_images)}")
        else:
            for idx, p in enumerate(all_images):
                im = Image.open(p).convert("RGB")
                im_resized = im.resize((args.res, args.res), resample=Image.BICUBIC)
                adv_im = pil_downup(
                    im_resized,
                    scale_w=float(args.scale_w),
                    scale_h=float(args.scale_h),
                    down_filter=args.down_filter,
                    up_filter=args.up_filter,
                )

                origin_path = save_dir / f"{idx:04d}_originImage.png"
                adv_path = save_dir / f"{idx:04d}_adv_image.png"
                im_resized.save(origin_path)
                adv_im.save(adv_path)

                # Predictions for stats
                x = tfm(im_resized).unsqueeze(0).to(device)
                x_adv = tfm(adv_im).unsqueeze(0).to(device)
                x_norm = normalize(x)
                x_adv_norm = normalize(x_adv)
                for di, det in enumerate(detectors):
                    pc = int(predict_label(det, x_norm).item() == 0)
                    pa = int(predict_label(det, x_adv_norm).item() == 0)
                    clean_correct[di] += pc
                    adv_correct[di] += pa

                # Metrics
                a = np.array(im_resized).astype(np.uint8)
                b = np.array(adv_im).astype(np.uint8)
                psnrs.append(calculate_psnr(a, b))
                try:
                    ssims.append(calculate_ssim(a, b))
                except Exception:
                    pass

                subset = infer_subset_from_path(p)
                w.writerow([idx, str(p), subset, str(origin_path), str(adv_path)])
                if (idx + 1) % 50 == 0 or (idx + 1) == len(all_images):
                    logger.info(f"processed {idx+1}/{len(all_images)}")

    N = len(all_images)
    logger.info(f"manifest={manifest}")
    logger.info(f"Mean PSNR={float(np.mean(psnrs)):.2f}  Std PSNR={float(np.std(psnrs)):.2f}")
    if ssims:
        logger.info(f"Mean SSIM={float(np.mean(ssims)):.4f}  Std SSIM={float(np.std(ssims)):.4f}")
    for i, n in enumerate(names):
        clean_acc = clean_correct[i] / max(N, 1)
        adv_acc = adv_correct[i] / max(N, 1)
        asr = 1.0 - adv_acc
        logger.info(f"[{n}] clean_acc={clean_acc*100:.2f}% adv_acc={adv_acc*100:.2f}% ASR(fake->real)={asr*100:.2f}%")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

