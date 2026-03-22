import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def calculate_mad(img1, img2):
    return float(np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16))))


def save_uint8(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


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


def build_image_cond(pipe, lowres_image, dtype, device):
    image = pipe.image_processor.preprocess(lowres_image)
    image = image.to(dtype=dtype, device=device)
    if image.shape[1] == 3:
        image = pipe.vae.encode(image).latent_dist.sample() * pipe.vae.config.scaling_factor
    image_cond = F.interpolate(image, scale_factor=2, mode="nearest")
    return image_cond


def build_text_condition(pipe, prompt, device):
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_length=True,
        return_tensors="pt",
    )
    text_encoder_out = pipe.text_encoder(
        text_inputs.input_ids.to(device),
        output_hidden_states=True,
    )
    text_embeddings = text_encoder_out.hidden_states[-1]
    text_pooler_out = text_encoder_out.pooler_output
    noise_level_embed = torch.cat(
        [
            torch.ones(text_pooler_out.shape[0], 64, dtype=text_pooler_out.dtype, device=device),
            torch.zeros(text_pooler_out.shape[0], 64, dtype=text_pooler_out.dtype, device=device),
        ],
        dim=1,
    )
    timestep_condition = torch.cat([noise_level_embed, text_pooler_out], dim=1)
    return text_embeddings, timestep_condition


def denoise_from_latent(pipe, zt, image_cond, text_embeddings, timestep_condition, timesteps, start_index):
    z = zt
    for i, t in enumerate(timesteps):
        sigma = pipe.scheduler.sigmas[i + start_index]
        scaled_model_input = pipe.scheduler.scale_model_input(z, t)
        model_input = torch.cat([scaled_model_input, image_cond], dim=1)
        timestep = torch.log(sigma) * 0.25
        raw_pred = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            timestep_cond=timestep_condition,
        ).sample
        raw_pred = raw_pred[:, :-1]
        inv_sigma = 1 / (sigma**2 + 1)
        model_pred = inv_sigma * z + pipe.scheduler.scale_model_input(sigma, t) * raw_pred
        z = pipe.scheduler.step(model_pred, t, z).prev_sample
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/gpufree-data/checkpoints/hf_models/sd-x2-latent-upscaler")
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--t_start", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_euler_set_timesteps()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)

    clean_pil = Image.open(args.image_path).convert("RGB")
    if clean_pil.size != (256, 256):
        clean_pil = clean_pil.resize((256, 256), Image.LANCZOS)
    clean_np = np.array(clean_pil, dtype=np.uint8)
    save_uint8(clean_np, output_dir / "clean_256.png")

    lowres_pil = clean_pil.resize((128, 128), Image.LANCZOS)
    save_uint8(np.array(lowres_pil, dtype=np.uint8), output_dir / "input_128.png")

    clean_tensor = pipe.image_processor.preprocess(clean_pil).to(device=device, dtype=dtype)
    with torch.no_grad():
        z0 = pipe.vae.encode(clean_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        image_cond = build_image_cond(pipe, lowres_pil, dtype=dtype, device=device)
        text_embeddings, timestep_condition = build_text_condition(pipe, [""], device)

        timesteps = pipe.scheduler.timesteps
        start_index = int(torch.argmin(torch.abs(timesteps - args.t_start)).item())
        chosen_t = timesteps[start_index]
        active_timesteps = timesteps[start_index:]

        noise = torch.randn_like(z0)
        zt = pipe.scheduler.add_noise(z0, noise, chosen_t.unsqueeze(0))

        z_final = denoise_from_latent(
            pipe=pipe,
            zt=zt,
            image_cond=image_cond,
            text_embeddings=text_embeddings,
            timestep_condition=timestep_condition,
            timesteps=active_timesteps,
            start_index=start_index,
        )
        recon = pipe.vae.decode(z_final / pipe.vae.config.scaling_factor, return_dict=False)[0]
        recon = pipe.image_processor.postprocess(recon, output_type="pil")[0]
        recon_np = np.array(recon.convert("RGB"), dtype=np.uint8)
        save_uint8(recon_np, output_dir / "recon_add_noise_euler.png")

    metrics = {
        "image_path": args.image_path,
        "model_path": args.model_path,
        "device": device,
        "num_inference_steps": args.num_inference_steps,
        "requested_t_start": args.t_start,
        "chosen_timestep_index": start_index,
        "chosen_timestep_value": float(chosen_t.item()),
        "num_active_steps": int(len(active_timesteps)),
        "psnr_to_clean": calculate_psnr(recon_np, clean_np),
        "mad_to_clean": calculate_mad(recon_np, clean_np),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
