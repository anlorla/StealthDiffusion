#!/usr/bin/env python3
"""
Run DiffAttack (from /root/gpufree-data/DiffAttack) as a baseline on our GenImage fake subset.

Why a wrapper script?
- The upstream DiffAttack repo expects ImageNet(-compatible) labels + its own utils/distances modules.
- Our project has name collisions (utils.py, distances.py, other_attacks.py).
- This script imports DiffAttack code while injecting *our* detector models (E/R/D/S) as `other_attacks`.

Outputs are compatible with our evaluation pipeline:
  <save_dir>/0000_originImage.png
  <save_dir>/0000_adv_image.png
  ...
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
from natsort import natsorted, ns
from PIL import Image

from loggers import Logger


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(images_root: str) -> list[str]:
    root = Path(images_root)
    if root.is_file():
        return [str(root)]
    paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return natsorted(paths, alg=ns.PATH)


def inject_binary_imagenet_label() -> None:
    """
    DiffAttack uses `from dataset_caption import imagenet_label` when dataset_name=imagenet_compatible.
    We inject a minimal 2-class mapping so prompts are meaningful for forensic setting.
    """
    pkg = types.ModuleType("dataset_caption")
    pkg.__path__ = []  # mark as package
    mod = types.ModuleType("dataset_caption.imagenet_label")
    mod.Label = {0: "ai", 1: "nature"}
    mod.refined_Label = {0: "ai", 1: "nature"}
    pkg.imagenet_label = mod
    sys.modules["dataset_caption"] = pkg
    sys.modules["dataset_caption.imagenet_label"] = mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, help="Fake images root (can contain subset subfolders)")
    ap.add_argument("--save_dir", required=True, help="Output folder")
    ap.add_argument("--model_name", default="S", help="Surrogate detector name: one of {S,E,R,D}")
    ap.add_argument("--pretrained_diffusion_path", default="stabilityai/stable-diffusion-2-1-base")
    ap.add_argument("--diffusion_steps", type=int, default=20)
    ap.add_argument("--start_step", type=int, default=18)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--attack_loss_weight", type=int, default=10)
    ap.add_argument("--cross_attn_loss_weight", type=int, default=10000)
    ap.add_argument("--self_attn_loss_weight", type=int, default=100)
    ap.add_argument("--self_replace_steps", type=float, default=1.0)
    ap.add_argument("--num_images", type=int, default=0, help=">0: only first N images")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(name="diffattack_baseline", log_path=str(save_dir / "diffattack_baseline.log"))

    # 1) Import our detectors as `other_attacks` for DiffAttack to use.
    # Ensure our repo root is importable before adding DiffAttack path.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    import other_attacks as our_other_attacks  # noqa: E402

    # 2) Put DiffAttack repo at the front so its `utils.py`, `distances.py`, etc are used.
    diffattack_root = Path("/root/gpufree-data/DiffAttack")
    sys.path.insert(0, str(diffattack_root))

    # 3) Inject modules to avoid name collisions / change behavior.
    sys.modules["other_attacks"] = our_other_attacks
    inject_binary_imagenet_label()

    # Now we can import DiffAttack modules safely.
    from diffusers import StableDiffusionPipeline, DDIMScheduler  # noqa: E402
    from attentionControl import AttentionControlEdit  # noqa: E402
    import diff_latent_attack as diffattack_impl  # noqa: E402

    images = collect_images(args.images_root)
    if args.num_images and args.num_images > 0:
        images = images[: args.num_images]
    if not images:
        raise FileNotFoundError(f"No images under {args.images_root!r}")

    # Fake-only baseline: label=0 for all images.
    labels = np.zeros((len(images),), dtype=np.int64)

    logger.info(f"images_root={args.images_root} count={len(images)}")
    logger.info(f"surrogate={args.model_name}")
    logger.info(
        f"diffusion_steps={args.diffusion_steps} start_step={args.start_step} iterations={args.iterations} "
        f"res={args.res} guidance={args.guidance}"
    )
    logger.info(
        f"weights: attack={args.attack_loss_weight} cross_attn={args.cross_attn_loss_weight} self_attn={args.self_attn_loss_weight}"
    )

    # Build diffusion pipeline
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path).to("cuda:0")
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    # Build a lightweight args object consumed by DiffAttack implementation
    diff_args = types.SimpleNamespace(
        dataset_name="imagenet_compatible",
        res=args.res,
        is_apply_mask=False,
        is_hard_mask=False,
        attack_loss_weight=args.attack_loss_weight,
        cross_attn_loss_weight=args.cross_attn_loss_weight,
        self_attn_loss_weight=args.self_attn_loss_weight,
    )

    # Run
    for idx, image_path in enumerate(images):
        img = Image.open(image_path).convert("RGB")
        Image.Image.save(img, save_dir / f"{idx:04d}_originImage.png")

        controller = AttentionControlEdit(args.diffusion_steps, args.self_replace_steps, args.res)
        # DiffAttack will save to: save_path + "_adv_image.png"
        save_path = str(save_dir / f"{idx:04d}")

        _adv_img, clean_acc, adv_acc = diffattack_impl.diffattack(
            ldm_stable,
            labels[idx : idx + 1],
            controller,
            num_inference_steps=args.diffusion_steps,
            guidance_scale=args.guidance,
            image=img,
            model_name=args.model_name,
            save_path=save_path,
            res=args.res,
            start_step=args.start_step,
            iterations=args.iterations,
            args=diff_args,
        )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(images):
            logger.info(f"[{idx+1}/{len(images)}] clean_acc={clean_acc} adv_acc={adv_acc}")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

