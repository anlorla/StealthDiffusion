import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionLatentUpscalePipeline
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from PIL import Image


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def patch_euler_set_timesteps():
    original = EulerDiscreteScheduler.set_timesteps

    def patched(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(
                0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.float32
            )[::-1].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
            timesteps -= 1
        else:
            return original(self, num_inference_steps, device=device)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.interpolation_type == "linear":
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        elif self.config.interpolation_type == "log_linear":
            sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp()
        else:
            return original(self, num_inference_steps, device=device)

        if self.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

        if isinstance(sigmas, np.ndarray):
            sigmas = torch.from_numpy(sigmas)
        sigmas = sigmas.to(dtype=torch.float32, device=device)

        if self.config.timestep_type == "continuous" and self.config.prediction_type == "v_prediction":
            self.timesteps = torch.tensor([0.25 * sigma.log() for sigma in sigmas], device=device)
        else:
            self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self._step_index = None

    EulerDiscreteScheduler.set_timesteps = patched


def run_once(pipe, clean_image, lowres_image, output_dir, guidance_scale, num_inference_steps):
    result = pipe(
        prompt="",
        image=lowres_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    out = result.images[0]
    out_path = output_dir / f"official_output_g{guidance_scale:g}.png"
    out.save(out_path)

    clean_np = np.array(clean_image.convert("RGB"), dtype=np.uint8)
    out_np = np.array(out.convert("RGB"), dtype=np.uint8)
    return {
        "guidance_scale": guidance_scale,
        "output_path": str(out_path),
        "psnr_to_clean": calculate_psnr(out_np, clean_np),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/gpufree-data/checkpoints/hf_models/sd-x2-latent-upscaler")
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_euler_set_timesteps()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    clean_image = Image.open(args.image_path).convert("RGB")
    if clean_image.size != (256, 256):
        clean_image = clean_image.resize((256, 256), Image.LANCZOS)
    lowres_image = clean_image.resize((128, 128), Image.LANCZOS)

    clean_image.save(output_dir / "clean_256.png")
    lowres_image.save(output_dir / "input_128.png")

    metrics = {
        "image_path": args.image_path,
        "model_path": args.model_path,
        "device": device,
        "num_inference_steps": args.num_inference_steps,
        "runs": [],
    }
    for guidance_scale in [0.0, 9.0]:
        metrics["runs"].append(
            run_once(
                pipe=pipe,
                clean_image=clean_image,
                lowres_image=lowres_image,
                output_dir=output_dir,
                guidance_scale=guidance_scale,
                num_inference_steps=args.num_inference_steps,
            )
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
