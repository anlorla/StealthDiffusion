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


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(images_root: str) -> list[str]:
    root = Path(images_root)
    if root.is_file():
        return [str(root)]
    paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return natsorted(paths, alg=ns.PATH)


def _maybe_inject_binary_imagenet_label() -> None:
    """
    Compatibility fallback.

    If DiffAttack doesn't support our 2-class dataset_caption (ours_try/ours_label),
    we can still run with dataset_name=imagenet_compatible but inject a 2-class imagenet_label
    so the prompt construction remains meaningful for our forensic setting.
    """
    pkg = types.ModuleType("dataset_caption")
    pkg.__path__ = []  # mark as package
    mod = types.ModuleType("dataset_caption.imagenet_label")
    mod.Label = {0: "ai", 1: "nature"}
    mod.refined_Label = {0: "ai", 1: "nature"}
    pkg.imagenet_label = mod
    sys.modules.setdefault("dataset_caption", pkg)
    sys.modules["dataset_caption.imagenet_label"] = mod


def load_labels(
    images: list[str],
    *,
    label_path: str | None,
    dataset_name: str,
) -> np.ndarray:
    """
    Keep the label semantics consistent with DiffAttack/main.py:
    - imagenet_compatible labels are 1-based in files and need -1
    - ours_try labels are already 0/1
    """
    if not label_path:
        # Fake-only baseline: label=0 (ai) for all images.
        return np.zeros((len(images),), dtype=np.int64)

    p = Path(label_path)
    raw: list[int] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        raw.append(int(s))

    if len(raw) != len(images):
        raise ValueError(f"label count ({len(raw)}) != image count ({len(images)}). label_path={label_path!r}")

    if dataset_name == "ours_try":
        return np.asarray(raw, dtype=np.int64)
    return (np.asarray(raw, dtype=np.int64) - 1).astype(np.int64)


def resolve_diffattack_root(repo_root: Path, diffattack_root_arg: str | None) -> Path:
    # Priority: CLI arg > env var > vendored path > legacy absolute path.
    if diffattack_root_arg:
        return Path(diffattack_root_arg)
    env = os.environ.get("DIFFATTACK_ROOT")
    if env:
        return Path(env)
    vendored = repo_root / "third_party" / "DiffAttack"
    if vendored.exists():
        return vendored
    return Path("/root/gpufree-data/DiffAttack")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, help="Fake images root (can contain subset subfolders)")
    ap.add_argument("--save_dir", required=True, help="Output folder")
    ap.add_argument("--model_name", default="S", help="Surrogate detector name: one of {S,E,R,D}")
    ap.add_argument(
        "--dataset_name",
        default="ours_try",
        choices=["ours_try", "imagenet_compatible", "cub_200_2011", "standford_car"],
        help="DiffAttack dataset_name; use ours_try for our 2-class (ai/nature) forensic setting",
    )
    ap.add_argument(
        "--label_path",
        default=None,
        help="Optional labels file aligned with the sorted image list. If omitted, assume all label=0 (fake/ai).",
    )
    ap.add_argument(
        "--diffattack_root",
        default=None,
        help="Path to DiffAttack repo. If omitted: $DIFFATTACK_ROOT or ./third_party/DiffAttack or /root/gpufree-data/DiffAttack",
    )
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

    # Ensure repo root is importable even when invoked as `python scripts/xxx.py`.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from loggers import Logger  # noqa: E402

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(name="diffattack_baseline", log_path=str(save_dir / "diffattack_baseline.log"))

    # 1) Import our detectors as `other_attacks` for DiffAttack to use.
    # Ensure our repo root is importable before adding DiffAttack path.
    import other_attacks as our_other_attacks  # noqa: E402

    # 2) Put DiffAttack repo at the front so its `utils.py`, `distances.py`, etc are used.
    diffattack_root = resolve_diffattack_root(repo_root, args.diffattack_root)
    sys.path.insert(0, str(diffattack_root))

    # 3) Inject modules to avoid name collisions / change behavior.
    sys.modules["other_attacks"] = our_other_attacks

    # Now we can import DiffAttack modules safely.
    from diffusers import StableDiffusionPipeline, DDIMScheduler  # noqa: E402
    from attentionControl import AttentionControlEdit  # noqa: E402
    import diff_latent_attack as diffattack_impl  # noqa: E402

    images = collect_images(args.images_root)
    if args.num_images and args.num_images > 0:
        images = images[: args.num_images]
    if not images:
        raise FileNotFoundError(f"No images under {args.images_root!r}")

    # Prefer keeping DiffAttack logic intact: use its dataset_caption/{ours_label,imagenet_label,...}.
    # If our customized "ours_try" isn't available, fall back to imagenet_compatible with a 2-class prompt mapping.
    dataset_name = args.dataset_name
    if dataset_name == "ours_try":
        impl_path = diffattack_root / "diff_latent_attack.py"
        if impl_path.exists():
            src = impl_path.read_text(errors="ignore")
            if "ours_try" not in src:
                logger.warning(
                    "DiffAttack repo doesn't support dataset_name=ours_try; falling back to imagenet_compatible."
                )
                dataset_name = "imagenet_compatible"
                _maybe_inject_binary_imagenet_label()

    labels = load_labels(images, label_path=args.label_path, dataset_name=dataset_name)

    logger.info(f"images_root={args.images_root} count={len(images)}")
    logger.info(f"surrogate={args.model_name}")
    logger.info(f"dataset_name={dataset_name} label_path={args.label_path}")
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
        dataset_name=dataset_name,
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
        img.save(save_dir / f"{idx:04d}_originImage.png")

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
