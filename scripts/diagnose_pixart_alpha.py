#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler, PixArtAlphaPipeline, StableDiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diff_latent_attack  # noqa: E402
from main import load_diffusion_pipeline  # noqa: E402


DEFAULT_SD_PATH = Path(
    "/root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/"
    "5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06"
)
DEFAULT_PIXART_PATH = Path("/root/gpufree-data/checkpoints/hf_models/pixart-alpha-xl-2-512x512")
DEFAULT_IMAGE_PATH = Path(
    "/root/gpufree-data/dataset/original_genimage_eval_1400/fake/adm/385_adm_145.PNG"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "pixart_alpha_diagnostics"


def check_network(host: str = "huggingface.co", port: int = 443, timeout: float = 5.0) -> dict[str, object]:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return {"reachable": True, "host": host, "port": port}
    except OSError as exc:
        return {"reachable": False, "host": host, "port": port, "error": str(exc)}
    finally:
        sock.close()


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_note(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image_tensor = image_tensor.detach().float().cpu().clamp(-1.0, 1.0)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image = (image_tensor.permute(1, 2, 0) / 2 + 0.5).clamp(0, 1).numpy()
    return (image * 255.0).round().astype(np.uint8)


def save_tensor_image(image_tensor: torch.Tensor, path: Path) -> None:
    Image.fromarray(tensor_to_uint8(image_tensor)).save(path)


def load_image(path: Path, res: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((res, res), resample=Image.LANCZOS)


def maybe_enable_offload(pipe) -> str:
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return "model_cpu_offload"
    if hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        return "sequential_cpu_offload"
    return "none"


@torch.no_grad()
def run_vae_recon(pipe, image: Image.Image, res: int, out_path: Path) -> dict[str, float]:
    latents = diff_latent_attack.encoder(image, pipe, res=res)
    decode_latents = (latents / diff_latent_attack.get_scaling_factor(pipe)).to(dtype=next(pipe.vae.parameters()).dtype)
    recon = pipe.vae.decode(decode_latents)["sample"]
    save_tensor_image(recon, out_path)

    image_uint8 = np.array(image, dtype=np.uint8)
    recon_uint8 = tensor_to_uint8(recon)
    return {
        "psnr": float(diff_latent_attack.calculate_psnr(image_uint8, recon_uint8)),
        "ssim": float(diff_latent_attack.calculate_ssim(image_uint8, recon_uint8)),
    }


@torch.no_grad()
def run_ddim_recon(pipe, image: Image.Image, res: int, out_path: Path, num_inference_steps: int) -> dict[str, float]:
    latent, inversion_latents = diff_latent_attack.ddim_reverse_sample(
        image,
        [""],
        pipe,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        res=res,
    )
    latent = inversion_latents[-1]
    context = diff_latent_attack.build_context(pipe, [""], res=res)
    for t in pipe.scheduler.timesteps:
        latent = diff_latent_attack.diffusion_step(pipe, latent, context, t, guidance_scale=0.0)

    recon = pipe.vae.decode(latent / diff_latent_attack.get_scaling_factor(pipe))["sample"]
    save_tensor_image(recon, out_path)

    image_uint8 = np.array(image, dtype=np.uint8)
    recon_uint8 = tensor_to_uint8(recon)
    return {
        "psnr": float(diff_latent_attack.calculate_psnr(image_uint8, recon_uint8)),
        "ssim": float(diff_latent_attack.calculate_ssim(image_uint8, recon_uint8)),
    }


def run_memory_probe(pipe, res: int) -> dict[str, float]:
    denoiser = diff_latent_attack.get_denoiser(pipe)
    device = next(denoiser.parameters()).device
    dtype = next(denoiser.parameters()).dtype
    pipe.scheduler.set_timesteps(1, device=device)

    prompt_info = diff_latent_attack.encode_prompt_embeddings(pipe, [""])
    if diff_latent_attack.is_pixart_alpha_backbone(pipe):
        context = diff_latent_attack.build_context(pipe, [""], res=res)
    else:
        context = {"encoder_hidden_states": torch.cat([prompt_info["uncond_embeddings"], prompt_info["text_embeddings"]], dim=0)}

    latent_h = res // pipe.vae_scale_factor
    latent_w = res // pipe.vae_scale_factor
    latents = torch.randn(1, denoiser.config.in_channels, latent_h, latent_w, device=device, dtype=dtype, requires_grad=True)
    t = pipe.scheduler.timesteps[0]

    torch.cuda.reset_peak_memory_stats(device)
    pred = diff_latent_attack.get_model_prediction(pipe, latents, context, t, guidance_scale=0.0)
    loss = pred.float().square().mean()
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {"peak_allocated_mb": float(peak_mb), "loss": float(loss.detach().cpu())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixart_path", type=Path, default=DEFAULT_PIXART_PATH)
    ap.add_argument("--sd_path", type=Path, default=DEFAULT_SD_PATH)
    ap.add_argument("--image_path", type=Path, default=DEFAULT_IMAGE_PATH)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--prompt", default="a photo of a cat")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / f"run_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env_summary = {
        "timestamp": stamp,
        "python": sys.executable,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pixart_alpha_pipeline_import": PixArtAlphaPipeline is not None,
        "pixart_path": str(args.pixart_path),
        "pixart_path_exists": args.pixart_path.exists(),
        "sd_path": str(args.sd_path),
        "sd_path_exists": args.sd_path.exists(),
        "image_path": str(args.image_path),
        "image_path_exists": args.image_path.exists(),
        "network_check": check_network(),
    }
    save_json(out_dir / "env_summary.json", env_summary)

    note_lines = [
        "# PixArt-Alpha diagnostics",
        "",
        f"- out_dir: `{out_dir}`",
        f"- pixart_path_exists: `{args.pixart_path.exists()}`",
        f"- sd_path_exists: `{args.sd_path.exists()}`",
        f"- image_path_exists: `{args.image_path.exists()}`",
        f"- network_reachable: `{env_summary['network_check']['reachable']}`",
    ]

    if not args.image_path.exists() or not args.sd_path.exists():
        note_lines.extend(
            [
                "",
                "## Blocker",
                "- Required input image or local SD 2.1 checkpoint is missing.",
            ]
        )
        save_note(out_dir / "note.md", note_lines)
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary: dict[str, object] = {"env": env_summary}
    image = load_image(args.image_path, args.res)
    image.save(out_dir / "input_resized.png")

    sd = StableDiffusionPipeline.from_pretrained(str(args.sd_path), torch_dtype=torch.float16, local_files_only=True).to(device)
    sd.scheduler = DDIMScheduler.from_config(dict(sd.scheduler.config))
    summary["sd_vae_recon"] = run_vae_recon(sd, image, args.res, out_dir / "sd21_vae_recon.png")
    try:
        summary["sd_ddim_recon"] = run_ddim_recon(
            sd,
            image,
            args.res,
            out_dir / "sd21_ddim_recon.png",
            args.num_inference_steps,
        )
    except Exception as exc:
        summary["sd_ddim_recon"] = {"status": "failed", "error": repr(exc)}

    if not args.pixart_path.exists():
        summary["pixart_status"] = {
            "status": "blocked",
            "reason": "local checkpoint missing",
            "network_check": env_summary["network_check"],
        }
        save_json(out_dir / "summary.json", summary)
        note_lines.extend(
            [
                "",
                "## SD baseline available",
                f"- sd_vae_recon: `{summary.get('sd_vae_recon')}`",
                f"- sd_ddim_recon: `{summary.get('sd_ddim_recon')}`",
                "",
                "## Blocker",
                "- Local PixArt-Alpha checkpoint is missing, so native generation / VAE / DDIM diagnostics cannot run yet.",
                f"- Network probe for Hugging Face: `{env_summary['network_check']}`",
            ]
        )
        save_note(out_dir / "note.md", note_lines)
        return 0

    del sd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pixart = load_diffusion_pipeline(str(args.pixart_path), device)
    pixart.scheduler = DDIMScheduler.from_config(dict(pixart.scheduler.config))
    pixart.inverse_scheduler = DDIMInverseScheduler.from_config(dict(pixart.scheduler.config))
    summary["pixart_offload"] = maybe_enable_offload(pixart)

    try:
        result = pixart(
            prompt=args.prompt,
            negative_prompt="",
            num_inference_steps=min(args.num_inference_steps, 12),
            guidance_scale=4.5,
            height=args.res,
            width=args.res,
            output_type="pil",
            use_resolution_binning=False,
        )
        result.images[0].save(out_dir / "pixart_native_t2i.png")
        summary["native_t2i"] = {"status": "ok", "prompt": args.prompt}
    except Exception as exc:
        summary["native_t2i"] = {"status": "failed", "error": repr(exc)}

    pixart_vae = run_vae_recon(pixart, image, args.res, out_dir / "pixart_vae_recon.png")
    summary["pixart_vae_recon"] = pixart_vae

    try:
        summary["pixart_ddim_recon"] = run_ddim_recon(
            pixart,
            image,
            args.res,
            out_dir / "pixart_ddim_recon.png",
            args.num_inference_steps,
        )
    except Exception as exc:
        summary["pixart_ddim_recon"] = {"status": "failed", "error": repr(exc)}

    try:
        if torch.cuda.is_available():
            summary["pixart_memory_probe"] = run_memory_probe(pixart, args.res)
        else:
            summary["pixart_memory_probe"] = {"status": "skipped", "reason": "cuda unavailable"}
    except Exception as exc:
        summary["pixart_memory_probe"] = {"status": "failed", "error": repr(exc)}

    save_json(out_dir / "summary.json", summary)
    note_lines.extend(
        [
            "",
            "## Results",
            f"- native_t2i: `{summary.get('native_t2i')}`",
            f"- pixart_vae_recon: `{summary.get('pixart_vae_recon')}`",
            f"- sd_vae_recon: `{summary.get('sd_vae_recon')}`",
            f"- pixart_ddim_recon: `{summary.get('pixart_ddim_recon')}`",
            f"- pixart_memory_probe: `{summary.get('pixart_memory_probe')}`",
        ]
    )
    save_note(out_dir / "note.md", note_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
