from __future__ import annotations

import os
from loggers import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionLatentUpscalePipeline,
    DDIMScheduler,
    DDIMInverseScheduler,
    EulerDiscreteScheduler,
)
import diff_latent_attack
from PIL import Image
import numpy as np
import os
import glob
import csv
import shutil
import json
from datetime import datetime, timezone

import random

from natsort import ns, natsorted
import argparse
import other_attacks
from pathlib import Path
from typing import List

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir',
                    default="/root/gpufree-data/artifacts/StealthDiffusion_attack_out_{name}",
                    type=str, help='(LEGACY/flat) Where to save the index-based outputs like 0000_adv_image.png')
parser.add_argument(
    "--output_mode",
    default="final",
    choices=["final", "flat"],
    type=str,
    help="final: write a full dataroot (real/ + fake/<subset>/...) directly; flat: legacy index-based output",
)
parser.add_argument(
    "--clean_dataroot",
    default="/root/gpufree-data/dataset/original_genimage_eval_1400",
    type=str,
    help="Clean dataroot containing real/ and fake/ (used to build the final out_dataroot structure)",
)
parser.add_argument(
    "--out_dataroot",
    default="/root/gpufree-data/dataset/StealthDiffusion/attack_datasets/StealthDiffusion_original_genimage_eval_1400_adv_sur{surrogate}",
    type=str,
    help="(final) Output dataroot to create; supports {surrogate} and {name}",
)
parser.add_argument(
    "--real_mode",
    default="symlink",
    choices=["symlink", "copy"],
    type=str,
    help="(final) How to place clean real images under out_dataroot/real",
)
parser.add_argument(
    "--save_origin",
    default=0,
    type=int,
    help="(final) If 1, also save per-image *_originImage.png into out_dataroot/_origin (not needed for evaluation)",
)
parser.add_argument('--mode',
                    default="double",
                    type=str, help='')
parser.add_argument('--images_root',
                    default="/root/gpufree-data/dataset/original_genimage_eval_1400/fake",
                    type=str, help='The clean images root directory')
parser.add_argument('--pretrained_diffusion_path',
                    default="/root/gpufree-data/checkpoints/hf_models/sd-x2-latent-upscaler",
                    type=str,
                    help='Diffusion backbone path. Supports StableDiffusionPipeline and StableDiffusionLatentUpscalePipeline.')
parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
parser.add_argument('--start_step',
                    default=18,
                    type=int, help='Which DDIM step to start the attack')
parser.add_argument('--iterations',
                    default=10,
                    type=int, help='Iterations of optimizing the adv_image')
parser.add_argument('--lr',
                    default=5e-4,
                    type=float, help='Latent optimization learning rate')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--model_name',
                    default="E,R,D,S",
                    type=str, help='The surrogate model from which the adversarial examples are crafted')
parser.add_argument('--dataset_name',
                    default="ours_try",
                    type=str,
                    choices=["ours_try"],
                    help='The dataset name for generating adversarial examples')
parser.add_argument('--is_encoder', default=0, type=int)
parser.add_argument('--encoder_weights',
                    default="/root/gpufree-data/checkpoints/Controlvae.pt",
                    type=str)

parser.add_argument('--guidance', default=0., type=float, help='guidance scale of diffusion models')
parser.add_argument('--eps', default=4 / 255, type=float, help='guidance scale of diffusion models')
parser.add_argument('--attack_loss_weight', default=10, type=int, help='attack loss weight factor')
parser.add_argument('--pgd_steps', default=10, type=int, help='PGD steps for initializing perturbations')
parser.add_argument('--pgd_alpha', default=None, type=float, help='PGD step size (default: eps/pgd_steps)')
parser.add_argument('--lambda_l1', default=1.0, type=float, help='L1 loss coefficient (paper: alpha)')
parser.add_argument('--lambda_lpips', default=1.0, type=float, help='LPIPS loss coefficient (paper: beta)')
parser.add_argument('--lambda_perc', default=None, type=float, help='Alias of lambda_lpips for SR pilot runs')
parser.add_argument('--lambda_latent', default=0.0, type=float, help='Latent L1 regularization coefficient (not in paper; default 0)')
parser.add_argument('--lambda_f', default=0.0, type=float, help='Explicit frequency-alignment loss coefficient')
parser.add_argument('--attack_only_fake', default=1, type=int, help='If 1, only craft adversarial examples for fake images')
parser.add_argument('--max_per_source', default=0, type=int, help='If >0, sample up to N fake images per generator/source (paper: 100)')
parser.add_argument('--seed', default=42, type=int, help='Random seed used for sampling')
parser.add_argument('--t_start', default=100.0, type=float, help='SR backbone add_noise starting timestep value')
parser.add_argument(
    '--freqsep_input',
    action='store_true',
    help='SR diagnostic mode: feed low-pass-upsampled image into the SR latent path and recompose only the generated high-frequency residual',
)
parser.add_argument(
    '--freqsep_scale',
    default=2,
    type=int,
    help='Downsample scale used by --freqsep_input (default: 2 for x2 latent upscaler)',
)
parser.add_argument(
    '--freqsep_skip_recompose',
    action='store_true',
    help='SR diagnostic mode: use low-pass input for the SR latent path but do not recompose generated high-frequency residual back onto a base image',
)
parser.add_argument('--disable_pgd_warm_start', action='store_true', help='Skip PGD warm start and use the clean input image directly')
parser.add_argument('--save_intermediates', action='store_true', help='Save clean / PGD / inversion / final images for diagnosis')
parser.add_argument('--intermediate_root', default='', type=str, help='Root directory for intermediate images and per-image metadata')
parser.add_argument('--pseudo_label_csv', default='', type=str, help='Optional CSV with image_id/image_path/prompt_text for per-image pseudo labels')
parser.add_argument('--cross_attn_loss_weight', default=0.0, type=float, help='Cross-attention regularization loss weight (lambda_c)')
parser.add_argument('--cross_attn_layers', default='', type=str, help='Comma-separated cross-attention layer names or block ids')
parser.add_argument('--cross_attn_max_layers', default=5, type=int, help='Maximum number of representative cross-attention layers to regularize')
parser.add_argument('--self_attn_loss_weight', default=0.0, type=float, help='Self-attention preservation loss weight (gamma)')
parser.add_argument('--self_attn_layers', default='', type=str, help='Comma-separated self-attention layer names or block ids')
parser.add_argument('--self_attn_max_layers', default=5, type=int, help='Maximum number of representative self-attention layers to regularize')

def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(42)


def _fake_suffix(path: Path) -> str:
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if 'fake' not in lowered:
        return path.name
    idx = lowered.index('fake')
    return '/'.join(['fake', *parts[idx + 1 :]])


def load_pseudo_prompt_map(csv_path) -> dict[str, dict[str, str]]:
    if not csv_path:
        return {'by_image_id': {}, 'by_suffix': {}}

    by_image_id: dict[str, str] = {}
    by_suffix: dict[str, str] = {}
    with Path(csv_path).open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt_text'].strip()
            if not prompt:
                raise ValueError(f'Empty prompt_text row in {csv_path}: {row}')
            image_id = row['image_id'].strip()
            if image_id:
                by_image_id[image_id] = prompt
            image_path = row['image_path'].strip()
            if image_path:
                by_suffix[_fake_suffix(Path(image_path))] = prompt
    return {'by_image_id': by_image_id, 'by_suffix': by_suffix}


def resolve_prompt_text(image_path: str, prompt_map: dict[str, dict[str, str]]) -> str | None:
    by_suffix = prompt_map['by_suffix']
    by_image_id = prompt_map['by_image_id']
    path = Path(image_path)
    suffix = _fake_suffix(path)
    if suffix in by_suffix:
        return by_suffix[suffix]
    return by_image_id.get(path.stem)


def _infer_pipeline_class(pretrained_diffusion_path: str):
    model_index_path = Path(pretrained_diffusion_path) / "model_index.json"
    if model_index_path.exists():
        with open(model_index_path, "r") as f:
            model_index = json.load(f)
        class_name = model_index.get("_class_name", "")
        if class_name == "StableDiffusionLatentUpscalePipeline":
            return StableDiffusionLatentUpscalePipeline
        if class_name == "StableDiffusionPipeline":
            return StableDiffusionPipeline

    if "sd-x2-latent-upscaler" in str(pretrained_diffusion_path):
        return StableDiffusionLatentUpscalePipeline
    return StableDiffusionPipeline


def load_diffusion_pipeline(pretrained_diffusion_path: str, device: str):
    pipeline_cls = _infer_pipeline_class(pretrained_diffusion_path)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    from_pretrained_kwargs = {"torch_dtype": torch_dtype}
    if Path(pretrained_diffusion_path).exists():
        from_pretrained_kwargs["local_files_only"] = True

    pipe = pipeline_cls.from_pretrained(pretrained_diffusion_path, **from_pretrained_kwargs).to(device)
    if pipeline_cls is StableDiffusionLatentUpscalePipeline:
        diff_latent_attack.patch_euler_set_timesteps()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = None
    else:
        scheduler_config = dict(pipe.scheduler.config)
        pipe.scheduler = DDIMScheduler.from_config(scheduler_config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
    return pipe


def run_diffusion_attack(
    image,
    label,
    diffusion_model,
    diffusion_steps,
    guidance=2.5,
    out_path_adv=None,
    save_dir="",
    res=224,
    model_name="inception",
    start_step=15,
    iterations=30,
    classes=None,
    logger=None,
    args=None,
    intermediate_dir=None,
    image_path=None,
    prompt_text=None,
):
    # Note: the actual file save is handled inside diff_latent_attack.diffattack().
    attack_image = image
    if not diff_latent_attack.is_sr_backbone(diffusion_model):
        attack_image = image.resize((res, res), resample=Image.LANCZOS)
    adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnr, ssim = diff_latent_attack.diffattack(
        diffusion_model,
        label,
        num_inference_steps=diffusion_steps,
        guidance_scale=guidance,
        image=attack_image,
        compare=image,
        save_path=save_dir,
        out_path_adv=out_path_adv,
        res=res,
        model_name=model_name,
        start_step=start_step,
        classes=classes,
        iterations=iterations,
        logger=logger,
        args=args,
        idx=0,
        intermediate_dir=intermediate_dir,
        image_path=image_path,
        prompt_text=prompt_text,
    )
    adv_image = np.array(adv_image)
    return adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnr, ssim


if __name__ == "__main__":
    args = parser.parse_args()
    if args.lambda_perc is not None:
        args.lambda_lpips = args.lambda_perc
    prompt_map = load_pseudo_prompt_map(args.pseudo_label_csv)
    guidance = args.guidance
    diffusion_steps = args.diffusion_steps  # Total DDIM sampling steps.
    start_step = args.start_step  # Which DDIM step to start the attack.
    iterations = args.iterations  # Iterations of optimizing the adv_image.
    res = args.res  # Input image resized resolution.
    model_name = args.model_name.split(",")  # The surrogate model from which the adversarial examples are crafted.

    if args.dataset_name == "ours_try":
        assert model_name in [["E", "R", "D", "S"], ["R", "E", "D", "S"], ["D", "E", "R", "S"], ["S", "E", "R", "D"]], f"There is no pretrained weight of {model_name} for RealFake dataset."


    name = ""
    for model in model_name:
        name += model
    name += str(args.is_encoder)
    surrogate = model_name[0]

    # Logs go to exp/ (experiment records), not to the dataset folder.
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = Path(__file__).resolve().parent / "exp" / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(name="train", log_path=str(log_dir / f"attack_{name}_sur{surrogate}.log"))

    images_root = args.images_root  # The clean images' root directory.
    if args.save_intermediates and args.intermediate_root:
        intermediate_root = Path(args.intermediate_root)
        intermediate_root.mkdir(parents=True, exist_ok=True)
        with (intermediate_root / "run_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)


    logger.info(f"\n******Attack based on Diffusion, Attacked Dataset: {args.dataset_name}*********")

    # Change the path to "stabilityai/stable-diffusion-2-base" if you want to use the pretrained model.
    pretrained_diffusion_path = args.pretrained_diffusion_path

    ldm_stable = load_diffusion_pipeline(pretrained_diffusion_path, 'cuda')


    dic_ = {"E":"efficientnet-b0",
            "R":"resnet50",
            "D":"deit",
            "S":"swin-t",
            }

    # Collect image files. Datasets may store images under `fake/` and `real/` subfolders.
    # We support:
    # - pointing `--images_root` at a single class folder (labels inferred as constant)
    # - pointing `--images_root` at a parent folder containing both classes (labels inferred per-path)
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    root = Path(images_root)
    if root.is_file():
        all_images = [str(root)]
    else:
        # Recurse so `--images_root .../fake` works when images are stored under
        # `fake/<subset>/*.png` (GenImage-style layout).
        all_images = [
            str(p)
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in img_exts
        ]
    all_images = natsorted(all_images, alg=ns.PATH)
    if len(all_images) == 0:
        subdirs = []
        if root.is_dir():
            subdirs = [p.name for p in root.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"No image files found under images_root={images_root!r}. "
            "Point --images_root at a folder containing images (or a single image file). "
            + (f"Subdirectories found: {subdirs}. " if subdirs else "")
            + "If your dataset is split into subfolders, pass e.g. "
            "`--images_root /path/to/genimage/fake`."
        )

    # Infer labels from path components.
    # Convention used in this repo: `fake` -> 0 (ai), `real` -> 1 (nature).
    def _infer_label(p: str) -> int:
        parts = [x.lower() for x in Path(p).parts]
        # Accept folders like `fake/`, `fake_100x6/`, `fake_by_subset/` for convenience.
        has_fake = any(x == "fake" or x.startswith("fake") for x in parts)
        has_real = any(x == "real" or x.startswith("real") for x in parts)
        if has_fake and not has_real:
            return 0
        if has_real and not has_fake:
            return 1
        raise ValueError(
            f"Can't infer label from path {p!r}. Expected one of path components to be 'fake' or 'real'."
        )

    label = np.array([_infer_label(p) for p in all_images], dtype=np.int64)

    def _infer_source(p: str) -> str:
        """
        Best-effort generator/source inference for GenImage fake images.
        If filename contains one of known substrings, use it; otherwise fall back to a token.
        """
        name = Path(p).name.lower()
        # Keep this list small but practical for GenImage naming patterns.
        known = [
            "biggan",
            "gaugan",
            "glide",
            "ldm",
            "sdv4",
            "wukong",
            "midjourney",
            "adm",
            "vqdm",
        ]
        for k in known:
            if k in name:
                return k
        parts = name.split("_")
        return parts[1] if len(parts) >= 2 else "unknown"

    # Match paper protocol: craft adversarial examples for fake images (ai) and evaluate transfer to others.
    if args.attack_only_fake:
        keep = [i for i, y in enumerate(label) if int(y) == 0]
        all_images = [all_images[i] for i in keep]
        label = label[keep]

    # Optional: sample a fixed number of fake images per source/generator (paper: 100 per validation set).
    if args.max_per_source and args.max_per_source > 0:
        rng = random.Random(args.seed)
        groups = {}
        for p in all_images:
            groups.setdefault(_infer_source(p), []).append(p)
        sampled = []
        for k in sorted(groups.keys()):
            imgs = groups[k][:]
            rng.shuffle(imgs)
            sampled.extend(imgs[: args.max_per_source])
        all_images = natsorted(sampled, alg=ns.PATH)
        label = np.array([_infer_label(p) for p in all_images], dtype=np.int64)

    adv_images = []
    images = []
    adv_all_acc = 0
    adv_all_acc1 = 0
    adv_all_acc2 = 0
    adv_all_acc3 = 0
    per_image_records = []

    # s = 0
    psnrss = []
    ssimss = []
    model_name1, model_name2, model_name3, model_name4 = model_name

    classifier = other_attacks.model_selection(model_name1).eval()
    classifier.requires_grad_(False)

    classifier_supp = other_attacks.model_selection(model_name2).eval()
    classifier_supp.requires_grad_(False)

    classifier_supp1 = other_attacks.model_selection(model_name3).eval()
    classifier_supp1.requires_grad_(False)

    classifier_supp2 = other_attacks.model_selection(model_name4).eval()
    classifier_supp2.requires_grad_(False)

    classes = [classifier, classifier_supp, classifier_supp1, classifier_supp2]

    # Output handling:
    # - final: write a full dataroot structure directly under args.out_dataroot (no attack_out intermediate)
    # - flat:  legacy index-based output under args.save_dir
    output_mode = args.output_mode
    clean_dataroot = Path(args.clean_dataroot)

    def _format_path_template(tpl: str) -> str:
        if "{" in tpl and "}" in tpl:
            try:
                return tpl.format(name=name, surrogate=surrogate)
            except Exception:
                return tpl
        return tpl

    def _iter_images(root_dir: Path) -> List[Path]:
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        files: List[Path] = []
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in img_exts:
                files.append(p)
        files = natsorted([str(p) for p in files], alg=ns.PATH)
        return [Path(p) for p in files]

    def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
        if dst.exists() or dst.is_symlink():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            rel = os.path.relpath(src, start=dst.parent)
            os.symlink(rel, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown real_mode: {mode}")

    manifest_real = None
    manifest_fake = None
    manifest_fake_adv = None
    out_dataroot = None

    if output_mode == "final":
        out_dataroot = Path(_format_path_template(args.out_dataroot))
        if out_dataroot.exists() or out_dataroot.is_symlink():
            raise FileExistsError(f"out_dataroot already exists: {out_dataroot}")
        (out_dataroot / "real").mkdir(parents=True, exist_ok=True)
        (out_dataroot / "fake").mkdir(parents=True, exist_ok=True)

        # Real: link/copy from clean dataroot
        real_src = clean_dataroot / "real"
        real_files = _iter_images(real_src)
        manifest_real = out_dataroot / "manifest_real.csv"
        with manifest_real.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["src", "dst"])
            for p in real_files:
                src = p.resolve()
                dst = out_dataroot / "real" / p.name
                _link_or_copy(src, dst, args.real_mode)
                try:
                    w.writerow([str(src.relative_to(repo_root)), str(dst.relative_to(repo_root))])
                except Exception:
                    w.writerow([str(src), str(dst)])

        manifest_fake = out_dataroot / "manifest_fake.csv"
        manifest_fake_adv = out_dataroot / "manifest_fake_adv.csv"
        with manifest_fake.open("w", newline="") as f_fake, manifest_fake_adv.open("w", newline="") as f_adv:
            w_fake = csv.writer(f_fake)
            w_adv = csv.writer(f_adv)
            w_fake.writerow(["src", "dst"])
            w_adv.writerow(["idx", "clean_path", "adv_path", "out_path"])

            fake_root_clean = clean_dataroot / "fake"
            images_root_p = Path(images_root)

            for ind, image_path in enumerate(all_images):
                in_p = Path(image_path)
                tmp_image = Image.open(image_path).convert("RGB")
                prompt_text = resolve_prompt_text(str(in_p), prompt_map)
                if (args.cross_attn_loss_weight > 0 or args.pseudo_label_csv) and not prompt_text:
                    raise KeyError(f"Missing prompt_text for image_path={in_p}")

                # Compute output relative path under fake/.
                rel = None
                if fake_root_clean.exists():
                    try:
                        rel = in_p.relative_to(fake_root_clean)
                    except Exception:
                        rel = None
                if rel is None:
                    try:
                        rel = in_p.relative_to(images_root_p)
                    except Exception:
                        rel = Path(in_p.name)

                out_p = out_dataroot / "fake" / rel
                out_p.parent.mkdir(parents=True, exist_ok=True)
                intermediate_dir = None
                if args.save_intermediates and args.intermediate_root:
                    intermediate_dir = Path(args.intermediate_root) / rel
                    intermediate_dir.parent.mkdir(parents=True, exist_ok=True)

                if args.save_origin:
                    origin_p = out_dataroot / "_origin" / rel
                    origin_p.parent.mkdir(parents=True, exist_ok=True)
                    tmp_image.save(str(origin_p))

                adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnrv, ssimv = run_diffusion_attack(
                    tmp_image,
                    label[ind : ind + 1],
                    ldm_stable,
                    diffusion_steps,
                    guidance=guidance,
                    res=res,
                    model_name=model_name,
                    classes=classes,
                    start_step=start_step,
                    iterations=iterations,
                    logger=logger,
                    args=args,
                    out_path_adv=str(out_p),
                    save_dir="",
                    intermediate_dir=str(intermediate_dir) if intermediate_dir is not None else None,
                    image_path=str(in_p),
                    prompt_text=prompt_text,
                )

                adv_all_acc += adv_acc
                adv_all_acc1 += adv_acc1
                adv_all_acc2 += adv_acc2
                adv_all_acc3 += adv_acc3
                psnrss.append(psnrv)
                ssimss.append(ssimv)
                per_image_records.append(
                    {
                        "index": ind,
                        "image_path": str(in_p),
                        "adv_path": str(out_p),
                        "prompt_text": prompt_text or "",
                        "acc_main": float(adv_acc),
                        "acc_supp": float(adv_acc1),
                        "acc_supp1": float(adv_acc2),
                        "acc_supp2": float(adv_acc3),
                        "psnr": float(psnrv),
                        "ssim": float(ssimv),
                    }
                )
                logger.info("final PSNR: {:.2f} dB; final SSIM: {:.4f}.".format(psnrv, ssimv))

                clean_p = (fake_root_clean / rel) if fake_root_clean.exists() else in_p

                try:
                    clean_s = str(clean_p.relative_to(repo_root))
                except Exception:
                    clean_s = str(clean_p)
                try:
                    out_s = str(out_p.relative_to(repo_root))
                except Exception:
                    out_s = str(out_p)

                w_fake.writerow([out_s, out_s])
                w_adv.writerow([ind, clean_s, out_s, out_s])

        meta = out_dataroot / "meta.txt"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta.write_text(
            "\n".join(
                [
                    f"created_utc={now}",
                    f"baseline=StealthDiffusion",
                    f"surrogate={surrogate}",
                    f"mode=final",
                    f"clean_dataroot={clean_dataroot.resolve()}",
                    f"images_root={Path(images_root).resolve()}",
                    f"real_mode={args.real_mode}",
                    f"n_real={len(_iter_images(clean_dataroot / 'real')) if (clean_dataroot / 'real').exists() else 0}",
                    f"n_fake={len(all_images)}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    else:
        # Legacy flat output.
        save_dir = _format_path_template(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"flat save_dir={save_dir}")

        for ind, image_path in enumerate(all_images):
            tmp_image = Image.open(image_path).convert("RGB")
            prompt_text = resolve_prompt_text(str(Path(image_path)), prompt_map)
            if (args.cross_attn_loss_weight > 0 or args.pseudo_label_csv) and not prompt_text:
                raise KeyError(f"Missing prompt_text for image_path={image_path}")
            if args.save_origin:
                tmp_image.save(os.path.join(save_dir, str(ind).rjust(4, "0") + "_originImage.png"))

            out_p = os.path.join(save_dir, str(ind).rjust(4, "0") + "_adv_image.png")
            intermediate_dir = None
            if args.save_intermediates and args.intermediate_root:
                intermediate_dir = Path(args.intermediate_root) / str(ind).rjust(4, "0")
                intermediate_dir.parent.mkdir(parents=True, exist_ok=True)
            adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnrv, ssimv = run_diffusion_attack(
                tmp_image,
                label[ind : ind + 1],
                ldm_stable,
                diffusion_steps,
                guidance=guidance,
                res=res,
                model_name=model_name,
                classes=classes,
                start_step=start_step,
                iterations=iterations,
                logger=logger,
                save_dir="",
                args=args,
                out_path_adv=out_p,
                intermediate_dir=str(intermediate_dir) if intermediate_dir is not None else None,
                image_path=str(Path(image_path)),
                prompt_text=prompt_text,
            )

            adv_all_acc += adv_acc
            adv_all_acc1 += adv_acc1
            adv_all_acc2 += adv_acc2
            adv_all_acc3 += adv_acc3
            psnrss.append(psnrv)
            ssimss.append(ssimv)
            per_image_records.append(
                {
                    "index": ind,
                    "image_path": str(Path(image_path)),
                    "adv_path": str(out_p),
                    "prompt_text": prompt_text or "",
                    "acc_main": float(adv_acc),
                    "acc_supp": float(adv_acc1),
                    "acc_supp1": float(adv_acc2),
                    "acc_supp2": float(adv_acc3),
                    "psnr": float(psnrv),
                    "ssim": float(ssimv),
                }
            )
            logger.info("final PSNR: {:.2f} dB; final SSIM: {:.4f}.".format(psnrv, ssimv))

    summary = {
        "output_mode": output_mode,
        "num_images": len(all_images),
        "adv_acc": adv_all_acc / len(all_images) if all_images else 0.0,
        "adv_acc1": adv_all_acc1 / len(all_images) if all_images else 0.0,
        "adv_acc2": adv_all_acc2 / len(all_images) if all_images else 0.0,
        "adv_acc3": adv_all_acc3 / len(all_images) if all_images else 0.0,
        "mean_psnr": float(np.mean(psnrss)) if psnrss else 0.0,
        "mean_ssim": float(np.mean(ssimss)) if ssimss else 0.0,
        "std_psnr": float(np.std(psnrss)) if psnrss else 0.0,
        "std_ssim": float(np.std(ssimss)) if ssimss else 0.0,
        "args": vars(args),
    }

    if args.intermediate_root:
        intermediate_root = Path(args.intermediate_root)
        intermediate_root.mkdir(parents=True, exist_ok=True)
        with (intermediate_root / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with (intermediate_root / "per_image_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(per_image_records, f, indent=2, ensure_ascii=False)
    if out_dataroot is not None:
        with (out_dataroot / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"output_mode={output_mode}")
    if out_dataroot is not None:
        logger.info(f"out_dataroot={out_dataroot}")
    logger.info("Adv acc: {}%".format(adv_all_acc / len(all_images) * 100))
    logger.info("Adv acc1: {}%".format(adv_all_acc1 / len(all_images) * 100))
    logger.info("Adv acc2: {}%".format(adv_all_acc2 / len(all_images) * 100))
    logger.info("Adv acc3: {}%".format(adv_all_acc3 / len(all_images) * 100))
    logger.info('mean PSNR: {:.2f} dB; mean SSIM: {:.4f}.'.format(np.mean(psnrss), np.mean(ssimss)))
    logger.info('std PSNR: {:.2f} dB; std SSIM: {:.4f}.'.format(np.std(psnrss), np.std(ssimss)))
