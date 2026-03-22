import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import diff_latent_attack as dla
import main as sr_main


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_mad(img1, img2):
    return float(np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16))))


def save_uint8(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/gpufree-data/checkpoints/hf_models/sd-x2-latent-upscaler")
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument(
        "--indices",
        default="-2,-3,-4",
        help="Comma-separated indices into scheduler.timesteps, e.g. -2,-3,-4",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pipe = sr_main.load_diffusion_pipeline(args.model_path, "cuda")
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=pipe.device)

    clean_pil = Image.open(args.image_path).convert("RGB")
    if clean_pil.size != (256, 256):
        clean_pil = clean_pil.resize((256, 256), Image.LANCZOS)
    clean_np = np.array(clean_pil, dtype=np.uint8)
    save_uint8(clean_np, output_dir / "clean_256.png")

    vae_dtype = next(pipe.vae.parameters()).dtype
    init_image = dla.preprocess(clean_pil, 256, device=pipe.device, dtype=vae_dtype)
    init_mask = torch.ones((1, 1, 256, 256), device=pipe.device, dtype=vae_dtype)

    with torch.no_grad():
        z0 = dla.encode_image_to_latents(clean_pil, pipe, res=256).to(device=pipe.device)
        ctx = dla.build_context(pipe, [""], image=clean_pil, res=256)

        metrics = {
            "image_path": args.image_path,
            "model_path": args.model_path,
            "num_inference_steps": args.num_inference_steps,
            "indices": args.indices,
            "seed": args.seed,
            "runs": [],
        }

        for idx_str in [x.strip() for x in args.indices.split(",") if x.strip()]:
            idx = int(idx_str)
            timestep = pipe.scheduler.timesteps[idx]
            noise = torch.randn_like(z0)
            zt = pipe.scheduler.add_noise(z0, noise, timestep)
            model_output = dla.get_model_prediction(pipe, zt, ctx, timestep, guidance_scale=0.0)
            z_prev = pipe.scheduler.step(model_output, timestep, zt).prev_sample
            out_img, _ = dla.decode_latent_batch(
                pipe,
                torch.cat([z_prev, z_prev], dim=0),
                init_image,
                init_mask,
                256,
                256,
                vae_dtype,
            )
            out_np = dla.tensor_image_to_uint8(out_img)
            out_name = f"ddim_one_step_idx{idx}_t{int(timestep.item())}.png"
            save_uint8(out_np, output_dir / out_name)
            run = {
                "index": idx,
                "timestep": int(timestep.item()),
                "output_path": str(output_dir / out_name),
                "psnr_to_clean": calculate_psnr(out_np, clean_np),
                "mad_to_clean": calculate_mad(out_np, clean_np),
            }
            metrics["runs"].append(run)
            print(json.dumps(run, ensure_ascii=False))

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
