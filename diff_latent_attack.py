from __future__ import annotations

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import cv2
import math
from lpips import LPIPS
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.models.autoencoder_kl import (
    AutoencoderKL,
    Encoder,
    Decoder,
    AutoencoderKLOutput,
    DiagonalGaussianDistribution,
    DecoderOutput,
)
from ControlVAE import NewEncoder, NewDecoder, NewAutoencoderKL
import torchattacks
from cross_attn_regularizer import CrossAttentionRegularizer
from self_attn_regularizer import SelfAttentionRegularizer


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    h, w = img1.shape[:2]
    img1 = img1[border : h - border, border : w - border]
    img2 = img2[border : h - border, border : w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    """
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    h, w = img1.shape[:2]
    img1 = img1[border : h - border, border : w - border]
    img2 = img2[border : h - border, border : w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
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
    return ssim_map.mean()


# --------------------------------------------
# 预处理 & 编码
# --------------------------------------------
def preprocess(image, res=512, device=None, dtype=torch.float32):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :]

    if device is not None:
        image = image.to(device)
    image = image.to(dtype)

    return 2.0 * image - 1.0


def patch_euler_set_timesteps():
    original = EulerDiscreteScheduler.set_timesteps
    if getattr(original, "_stealthdiffusion_sr_patched", False):
        return

    def patched(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(
                0,
                self.config.num_train_timesteps - 1,
                num_inference_steps,
                dtype=np.float32,
            )[::-1].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(np.float32)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = (
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)
                .round()
                .copy()
                .astype(np.float32)
            )
            timesteps -= 1
        else:
            return original(self, num_inference_steps, device=device)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.interpolation_type == "linear":
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        elif self.config.interpolation_type == "log_linear":
            sigmas = (
                torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1)
                .exp()
            )
        else:
            return original(self, num_inference_steps, device=device)

        if self.use_karras_sigmas:
            sigmas = self._convert_to_karras(
                in_sigmas=sigmas,
                num_inference_steps=self.num_inference_steps,
            )
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

        if isinstance(sigmas, np.ndarray):
            sigmas = torch.from_numpy(sigmas)
        sigmas = sigmas.to(dtype=torch.float32, device=device)

        if self.config.timestep_type == "continuous" and self.config.prediction_type == "v_prediction":
            self.timesteps = torch.tensor(
                [0.25 * sigma.log() for sigma in sigmas],
                device=device,
            )
        else:
            self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self._step_index = None

    patched._stealthdiffusion_sr_patched = True
    EulerDiscreteScheduler.set_timesteps = patched


def is_sr_backbone(model):
    return (
        model.__class__.__name__ == "StableDiffusionLatentUpscalePipeline"
        or getattr(model.unet.config, "in_channels", None) == 8
    )


def get_scaling_factor(model):
    return float(getattr(model.vae.config, "scaling_factor", 0.18215))


def encode_image_to_latents(image, model, res=512):
    vae_param = next(model.vae.parameters())
    unet_param = next(model.unet.parameters())
    vae_device = vae_param.device
    vae_dtype = vae_param.dtype
    unet_dtype = unet_param.dtype

    generator = torch.Generator(device=vae_device).manual_seed(8888)
    image = preprocess(image, res, device=vae_device, dtype=vae_dtype)

    gpu_generator = torch.Generator(device=vae_device)
    gpu_generator.manual_seed(generator.initial_seed())

    latents = get_scaling_factor(model) * model.vae.encode(image).latent_dist.sample(
        generator=gpu_generator
    )

    latents = latents.to(dtype=unet_dtype)
    return latents


def encode_prompt_embeddings(model, prompt):
    prompt = prompt if isinstance(prompt, list) else [prompt]
    max_length = model.tokenizer.model_max_length

    if is_sr_backbone(model):
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_output = model.text_encoder(
            text_input.input_ids.to(model.device),
            output_hidden_states=True,
        )
        uncond_input = model.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_output = model.text_encoder(
            uncond_input.input_ids.to(model.device),
            output_hidden_states=True,
        )
        return {
            "text_embeddings": text_output.hidden_states[-1],
            "text_pooler": text_output.pooler_output,
            "uncond_embeddings": uncond_output.hidden_states[-1],
            "uncond_pooler": uncond_output.pooler_output,
        }

    uncond_input = model.tokenizer(
        [""] * len(prompt),
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return {
        "text_embeddings": text_embeddings,
        "uncond_embeddings": uncond_embeddings,
    }


def build_sr_image_condition(image, model, res, batch_size):
    if res % 2 != 0:
        raise ValueError(f"SR backbone expects an even output resolution, got res={res}")

    low_res = max(res // 2, 1)
    cond_latents = encode_image_to_latents(image, model, res=low_res)
    cond_latents = F.interpolate(cond_latents, scale_factor=2, mode="nearest")
    cond_latents = cond_latents.to(dtype=next(model.unet.parameters()).dtype)
    return torch.cat([cond_latents] * (batch_size * 2), dim=0)


def build_sr_lowres_image(image, res):
    if res % 2 != 0:
        raise ValueError(f"SR backbone expects an even output resolution, got res={res}")
    low_res = max(res // 2, 1)
    return image.resize((low_res, low_res), resample=Image.LANCZOS)


def build_lowpass_upsampled_image(image, scale=2):
    if scale <= 1:
        return image.copy()
    width, height = image.size
    low_w = max(width // scale, 1)
    low_h = max(height // scale, 1)
    low = image.resize((low_w, low_h), resample=Image.LANCZOS)
    return low.resize((width, height), resample=Image.BICUBIC)


def lowpass_tensor(image_tensor, scale=2):
    if scale <= 1:
        return image_tensor
    h, w = image_tensor.shape[-2:]
    low_h = max(h // scale, 1)
    low_w = max(w // scale, 1)
    low = F.interpolate(
        image_tensor.float(),
        size=(low_h, low_w),
        mode="area",
    )
    up = F.interpolate(
        low,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    return up.to(dtype=image_tensor.dtype)


def build_lowpass_upsampled_tensor(
    image,
    res,
    scale=2,
    device=None,
    dtype=torch.float32,
):
    image_tensor = preprocess(image, res=res, device=device, dtype=dtype)
    image_tensor = torch.nan_to_num(
        image_tensor, nan=0.0, posinf=1.0, neginf=-1.0
    ).clamp(-1.0, 1.0)
    return lowpass_tensor(image_tensor, scale=scale)


def recompose_with_generated_highfreq(attack_image, lowpass_base, scale=2):
    attack_low = lowpass_tensor(attack_image, scale=scale)
    high = attack_image - attack_low
    return (lowpass_base + high).clamp(-1.0, 1.0)


def build_context(
    model,
    prompt,
    image=None,
    res=512,
    uncond_embeddings=None,
    uncond_pooler=None,
):
    prompt = prompt if isinstance(prompt, list) else [prompt]
    prompt_info = encode_prompt_embeddings(model, prompt)
    text_embeddings = prompt_info["text_embeddings"]
    if uncond_embeddings is None:
        uncond_embeddings = prompt_info["uncond_embeddings"]

    context = {
        "encoder_hidden_states": torch.cat([uncond_embeddings, text_embeddings], dim=0),
        "do_classifier_free_guidance": True,
    }

    if is_sr_backbone(model):
        text_pooler = prompt_info["text_pooler"]
        if uncond_pooler is None:
            uncond_pooler = prompt_info["uncond_pooler"]
        pooled = torch.cat([uncond_pooler, text_pooler], dim=0)
        noise_level_embed = torch.cat(
            [
                torch.ones(pooled.shape[0], 64, dtype=pooled.dtype, device=pooled.device),
                torch.zeros(pooled.shape[0], 64, dtype=pooled.dtype, device=pooled.device),
            ],
            dim=1,
        )
        context["timestep_condition"] = torch.cat([noise_level_embed, pooled], dim=1)
        context["image_cond"] = build_sr_image_condition(image, model, res=res, batch_size=len(prompt))
        context["uncond_pooler"] = uncond_pooler
        context["text_pooler"] = text_pooler

    return context


def resolve_prompt_text_for_attack(imagenet_label, label_value: int, prompt_text):
    base_prompt = (prompt_text or imagenet_label.refined_Label[label_value]).strip()
    if not base_prompt:
        raise ValueError('Resolved prompt_text is empty')
    return base_prompt


def resolve_prompt_token_indices(tokenizer, prompt_text: str):
    max_length = getattr(tokenizer, 'model_max_length', 77)
    token_kwargs = {
        'padding': 'max_length',
        'max_length': max_length,
        'truncation': True,
        'return_tensors': 'pt',
    }
    try:
        encoded = tokenizer([prompt_text], return_special_tokens_mask=True, **token_kwargs)
    except TypeError:
        encoded = tokenizer([prompt_text], **token_kwargs)

    input_ids = encoded['input_ids'][0]
    attention_mask = encoded.get('attention_mask', torch.ones_like(input_ids))[0]
    special_mask = encoded.get('special_tokens_mask')
    if special_mask is not None:
        special_mask = special_mask[0]

    token_indices = []
    for idx in range(int(input_ids.shape[0])):
        if int(attention_mask[idx].item()) == 0:
            continue
        if special_mask is not None and int(special_mask[idx].item()) == 1:
            continue
        token_indices.append(idx)

    if token_indices:
        return token_indices

    active = [idx for idx in range(int(input_ids.shape[0])) if int(attention_mask[idx].item()) == 1]
    if len(active) > 2:
        return active[1:-1]
    return active


def get_model_prediction(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)

    if is_sr_backbone(model):
        sigma_index = int(
            torch.argmin(torch.abs(model.scheduler.timesteps.float() - float(t))).item()
        )
        sigma = model.scheduler.sigmas[sigma_index].to(
            device=latents_input.device,
            dtype=latents_input.dtype,
        ).clamp(min=1e-6)
        timestep = (torch.log(sigma) * 0.25).to(
            device=latents_input.device,
            dtype=latents_input.dtype,
        )
        scaled_latents = model.scheduler.scale_model_input(latents_input, t)
        model_input = torch.cat([scaled_latents, context["image_cond"]], dim=1)
        raw_pred = model.unet(
            model_input,
            timestep,
            encoder_hidden_states=context["encoder_hidden_states"],
            timestep_cond=context["timestep_condition"],
        ).sample
        raw_pred = raw_pred[:, :-1]
        inv_sigma = 1 / (sigma**2 + 1)
        sigma_coeff = sigma / torch.sqrt(sigma**2 + 1)
        model_pred = inv_sigma * latents_input + sigma_coeff * raw_pred
    else:
        model_pred = model.unet(
            latents_input,
            t,
            encoder_hidden_states=context["encoder_hidden_states"],
        )["sample"]

    noise_pred_uncond, noise_prediction_text = model_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)


@torch.no_grad()
def ddim_reverse_sample(
    image,
    prompt,
    model,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    res=512,
):
    """
    ==========================================
    ============ DDIM Inversion ==============
    ==========================================
    """
    model.scheduler.set_timesteps(num_inference_steps)
    model.inverse_scheduler.set_timesteps(num_inference_steps)

    latents = encode_image_to_latents(image, model, res=res)
    timesteps = model.inverse_scheduler.timesteps
    context = build_context(model, [prompt[0]], image=image, res=res)

    all_latents = [latents]

    for t in tqdm(timesteps[1:], desc="DDIM_inverse"):
        model_output = get_model_prediction(model, latents, context, t, guidance_scale)
        latents = model.inverse_scheduler.step(model_output, t, latents)["prev_sample"]
        all_latents.append(latents)

    return latents, all_latents


def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(
        batch_size, model.vae.config.latent_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    model_output = get_model_prediction(model, latents, context, t, guidance_scale)
    latents = model.scheduler.step(model_output, t, latents)["prev_sample"]
    return latents


def reset_scheduler_state(model):
    if is_sr_backbone(model) and hasattr(model.scheduler, "_step_index"):
        model.scheduler._step_index = None


def get_sr_start_latent(
    image,
    model,
    prompt,
    num_inference_steps,
    diffusion_res,
    t_start,
):
    device = next(model.unet.parameters()).device
    unet_dtype = next(model.unet.parameters()).dtype

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps
    start_index = int(torch.argmin(torch.abs(timesteps.float() - float(t_start))).item())
    chosen_t = timesteps[start_index]
    active_timesteps = timesteps[start_index:]

    z0 = encode_image_to_latents(image, model, res=diffusion_res)
    z0 = z0.to(device=device, dtype=unet_dtype)
    noise = torch.randn_like(z0)
    zt = model.scheduler.add_noise(z0, noise, chosen_t.unsqueeze(0))

    prompt_info = encode_prompt_embeddings(model, prompt)
    return {
        "latent": zt,
        "z0": z0,
        "active_timesteps": active_timesteps,
        "start_index": start_index,
        "chosen_timestep": chosen_t,
        "prompt_info": prompt_info,
    }


def latent2image(vae, latents, args, decoder, down_features):
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    if args.is_encoder == 1:
        latents = latents / scaling_factor
        latents = vae.post_quant_conv(latents)
        image = decoder(latents, down_features)
    else:
        latents = latents / scaling_factor
        image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
    image = (image * 255).astype(np.uint8)
    return image


def save_uint8_image(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def pil_to_uint8(image):
    return np.array(image.convert("RGB"), dtype=np.uint8)


def neg1pos1_tensor_to_uint8(image_tensor):
    if image_tensor.ndim == 4:
        if image_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for image tensor, got shape={tuple(image_tensor.shape)}"
            )
        image_tensor = image_tensor[0]
    image_tensor = torch.nan_to_num(
        image_tensor.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=-1.0
    ).clamp(-1.0, 1.0)
    return np.clip(
        ((image_tensor + 1.0) * 127.5).round().permute(1, 2, 0).numpy(),
        0,
        255,
    ).astype(np.uint8)


def tensor_image_to_uint8(image_tensor):
    image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(
        image_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0, 0, 255
    ).astype(np.uint8)


def decode_latent_batch(model, latents, init_image, init_mask, diffusion_res, res, vae_dtype):
    decode_latents = (latents.detach() / get_scaling_factor(model)).to(dtype=vae_dtype)
    decode_latents = torch.nan_to_num(
        decode_latents, nan=0.0, posinf=5.0, neginf=-5.0
    )
    out_image = model.vae.decode(decode_latents)["sample"][1:] * init_mask + (
        1 - init_mask
    ) * init_image
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image_eval = out_image
    if diffusion_res != res:
        out_image_eval = F.interpolate(
            out_image.float(),
            size=(res, res),
            mode="bilinear",
            align_corners=False,
        ).to(out_image.dtype)
    return out_image, out_image_eval

@torch.enable_grad()
def diffattack(
    model,
    label,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    image=None,
    compare=None,
    model_name=["E", "R", "D", "S"],
    save_path="",
    res=224,
    start_step=15,
    classes=None,
    iterations=30,
    verbose=True,
    topN=1,
    logger=None,
    args=None,
    mode=None,
    idx=0,
    adm=None,
    out_path_adv=None,
    intermediate_dir=None,
    image_path=None,
    prompt_text=None,
):
    if args.dataset_name == "ours_try":
        from dataset_caption import ours_label as imagenet_label
    else:
        raise NotImplementedError

    # VAE / UNet 精度 & device
    unet_param = next(model.unet.parameters())
    vae_param = next(model.vae.parameters())
    device = unet_param.device
    unet_dtype = unet_param.dtype
    vae_dtype = vae_param.dtype

    label = torch.from_numpy(label).long().cuda()

    classifier, classifier_supp, classifier_supp1, classifier_supp2 = classes
    classifier = classifier.eval()
    classifier_supp = classifier_supp.eval()
    classifier_supp1 = classifier_supp1.eval()
    classifier_supp2 = classifier_supp2.eval()

    intermediate_dir_path = Path(intermediate_dir) if intermediate_dir else None
    if intermediate_dir_path is not None:
        intermediate_dir_path.mkdir(parents=True, exist_ok=True)
        meta = {"idx": idx}
        if image_path is not None:
            meta["image_path"] = image_path
        if prompt_text is not None:
            meta["prompt_text"] = prompt_text
        with (intermediate_dir_path / "meta.json").open("w", encoding="utf-8") as f:
            import json
            json.dump(meta, f, indent=2, ensure_ascii=False)

    height = width = res
    diffusion_res = res
    if is_sr_backbone(model):
        diffusion_res = int(math.ceil(res / 64.0) * 64)
        diffusion_res = max(diffusion_res, 256)
        logger.info(
            f"SR backbone uses diffusion_res={diffusion_res} while surrogate/eval stay at res={res}"
        )

    # ---- 预处理原始图像给分类器 ----
    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0).cuda()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )
    if intermediate_dir_path is not None:
        save_uint8_image(pil_to_uint8(compare), intermediate_dir_path / "clean.png")

    # ---- 先用 PGD 在图像域做一个初始扰动（paper: eps=4/255, iterations=10） ----
    pgd_eps = float(getattr(args, "eps", 4 / 255))
    pgd_steps = int(getattr(args, "pgd_steps", 10))
    pgd_alpha = getattr(args, "pgd_alpha", None)
    if pgd_alpha is None:
        pgd_alpha = pgd_eps / max(pgd_steps, 1)
    attack = torchattacks.PGD(
        classifier,
        eps=pgd_eps,
        alpha=float(pgd_alpha),
        steps=pgd_steps,
        random_start=False,
    )
    attack.set_normalization_used(mean=mean, std=std)

    if getattr(args, "disable_pgd_warm_start", False):
        pgd_uint8 = pil_to_uint8(image)
    else:
        adv_images = attack(test_image, label.detach().to(test_image.device))
        pgd_uint8 = np.uint8(
            denormalize(adv_images).detach().cpu().numpy()[0].transpose((1, 2, 0)) * 255.0
        )
        image = Image.fromarray(pgd_uint8)
    if intermediate_dir_path is not None:
        save_uint8_image(pgd_uint8, intermediate_dir_path / "pgd_warm_start.png")
        if is_sr_backbone(model):
            save_uint8_image(
                pil_to_uint8(build_sr_lowres_image(image, diffusion_res)),
                intermediate_dir_path / "input_128.png",
            )

    sr_operating_image = image
    freqsep_base_diffusion = None
    freqsep_base_compare = None
    freqsep_enabled = bool(
        is_sr_backbone(model) and getattr(args, "freqsep_input", False)
    )
    freqsep_scale = int(getattr(args, "freqsep_scale", 2))
    freqsep_recompose = not bool(getattr(args, "freqsep_skip_recompose", False))
    if freqsep_enabled:
        freqsep_source_image = compare if compare is not None else image
        freqsep_base_diffusion = build_lowpass_upsampled_tensor(
            freqsep_source_image,
            res=diffusion_res,
            scale=freqsep_scale,
            device=device,
            dtype=vae_dtype,
        )
        freqsep_base_compare = build_lowpass_upsampled_tensor(
            freqsep_source_image,
            res=res,
            scale=freqsep_scale,
            device=device,
            dtype=vae_dtype,
        )
        sr_operating_image = Image.fromarray(
            neg1pos1_tensor_to_uint8(freqsep_base_diffusion)
        )
        logger.info(
            f"[SR-FREQSEP] enabled with scale={freqsep_scale}; "
            "use clean compare image to build the tensor-space low-pass base for the SR latent path"
        )
        if freqsep_recompose:
            logger.info("[SR-FREQSEP] high-frequency recomposition is enabled")
        else:
            logger.info("[SR-FREQSEP] high-frequency recomposition is disabled")
        if intermediate_dir_path is not None:
            save_uint8_image(
                pil_to_uint8(sr_operating_image),
                intermediate_dir_path / "freqsep_lowpass_input.png",
            )

    # ---- 4 个分类器的 clean 性能（只看结果，不要梯度） ----
    with torch.no_grad():
        pred = classifier(test_image)
        pred1 = classifier_supp(test_image)
        pred2 = classifier_supp1(test_image)
        pred3 = classifier_supp2(test_image)

        pred_accuracy_clean = (
            (torch.argmax(pred, 1) == label).sum().item() / len(label)
        )
        pred_accuracy_clean1 = (
            (torch.argmax(pred1, 1) == label).sum().item() / len(label)
        )
        pred_accuracy_clean2 = (
            (torch.argmax(pred2, 1) == label).sum().item() / len(label)
        )
        pred_accuracy_clean3 = (
            (torch.argmax(pred3, 1) == label).sum().item() / len(label)
        )

    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))
    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean1 * 100))
    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean2 * 100))
    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean3 * 100))

    logit = torch.nn.Softmax(dim=1)(pred)
    logit1 = torch.nn.Softmax(dim=1)(pred1)
    logit2 = torch.nn.Softmax(dim=1)(pred2)
    logit3 = torch.nn.Softmax(dim=1)(pred3)
    logger.info(
        f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred, 1).item()}, "
        f"pred_clean_logit: {logit[0, label[0]].item()}"
    )
    logger.info(
        f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred1, 1).item()}, "
        f"pred_clean_logit: {logit1[0, label[0]].item()}"
    )
    logger.info(
        f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred2, 1).item()}, "
        f"pred_clean_logit: {logit2[0, label[0]].item()}"
    )
    logger.info(
        f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred3, 1).item()}, "
        f"pred_clean_logit: {logit3[0, label[0]].item()}"
    )

    # 这块其实没用 top-k 的 label，只是打印
    _, pred_labels = pred.topk(topN, largest=True, sorted=True)
    base_prompt = resolve_prompt_text_for_attack(
        imagenet_label,
        int(label[0].item()),
        prompt_text,
    )
    prompt = [base_prompt] * 2
    logger.info(
        f"prompt generate: {prompt[0]} \tlabels: {pred_labels.cpu().numpy().tolist()}"
    )

    prompt_token_indices = resolve_prompt_token_indices(model.tokenizer, base_prompt)
    if getattr(args, 'cross_attn_loss_weight', 0.0) > 0 and not prompt_token_indices:
        raise ValueError(f"No valid prompt token indices resolved for prompt_text={base_prompt!r}")
    if hasattr(model.tokenizer, 'encode'):
        decoder_tokens = model.tokenizer.encode(base_prompt)
        logger.info(f"decoder: {decoder_tokens}")
    logger.info(f"[PROMPT] base_prompt={base_prompt!r} token_indices={prompt_token_indices}")

    init_prompt = [base_prompt]

    batch_size_init = len(init_prompt)
    all_uncond_emb = []

    if is_sr_backbone(model):
        sr_t_start = float(getattr(args, "t_start", 100.0))
        sr_start = get_sr_start_latent(
            image=sr_operating_image,
            model=model,
            prompt=init_prompt,
            num_inference_steps=num_inference_steps,
            diffusion_res=diffusion_res,
            t_start=sr_t_start,
        )
        latent = sr_start["latent"]
        active_timesteps = sr_start["active_timesteps"]
        logger.info(
            f"[SR] requested t_start={sr_t_start:.4f}, "
            f"chosen index={sr_start['start_index']}, "
            f"chosen timestep={float(sr_start['chosen_timestep'].item()):.4f}, "
            f"active_steps={len(active_timesteps)}"
        )
        all_uncond_emb = [
            sr_start["prompt_info"]["uncond_embeddings"].detach().clone()
            for _ in range(len(active_timesteps))
        ]
        uncond_pooler = sr_start["prompt_info"].get("uncond_pooler")
    else:
        # ==========================================
        # ============ DDIM Inversion ==============
        # ==========================================
        latent, inversion_latents = ddim_reverse_sample(
            image,
            prompt,
            model,
            num_inference_steps,
            0,
            res=diffusion_res,
        )
        inversion_latents = inversion_latents[::-1]

        with torch.no_grad():
            for k, z in enumerate(inversion_latents):
                print(
                    f"[INV] step {k}: mean={z.mean().item():.4f}, std={z.std().item():.4f}, "
                    f"nan={torch.isnan(z).any().item()}, inf={torch.isinf(z).any().item()}"
                )

        latent = inversion_latents[start_step - 1]

        print(
            f"[INV] chosen latent (step {start_step-1}): "
            f"mean={latent.mean().item():.4f}, std={latent.std().item():.4f}, "
            f"nan={torch.isnan(latent).any().item()}, inf={torch.isinf(latent).any().item()}"
        )

        init_prompt_info = encode_prompt_embeddings(model, init_prompt)
        uncond_embeddings = init_prompt_info["uncond_embeddings"]
        uncond_pooler = init_prompt_info.get("uncond_pooler")

        latent, latents = init_latent(
            latent, model, diffusion_res, diffusion_res, batch_size_init
        )
        with torch.no_grad():
            for k, z in enumerate(latents):
                print(
                    f"[INV] step {k}: mean={z.mean().item():.4f}, std={z.std().item():.4f}, "
                    f"nan={torch.isnan(z).any().item()}, inf={torch.isinf(z).any().item()}"
                )

        uncond_embeddings.requires_grad_(True)

        active_timesteps = model.scheduler.timesteps[1 + start_step - 1 :]
        for _ind, _t in enumerate(
            tqdm(active_timesteps, desc="Optimize_uncond_embed")
        ):
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    # ==========================================
    # ============ Latents Attack ==============
    # ==========================================

    batch_size = len(prompt)

    prompt_info = encode_prompt_embeddings(model, prompt)
    text_embeddings = prompt_info["text_embeddings"]
    text_pooler = prompt_info.get("text_pooler")

    context = []
    for i in range(len(all_uncond_emb)):
        step_kwargs = {
            "model": model,
            "prompt": prompt,
            "image": sr_operating_image,
            "res": diffusion_res,
            "uncond_embeddings": all_uncond_emb[i].repeat(batch_size, 1, 1),
        }
        if text_pooler is not None and uncond_pooler is not None:
            step_kwargs["uncond_pooler"] = uncond_pooler.repeat(batch_size, 1)
        context.append(build_context(**step_kwargs))

    model.unet.requires_grad_(False)
    model.vae.requires_grad_(False)


    # inversion 得到的 latent（和 UNet 同 dtype，比如 float16），只用来做对比
    original_latent = latent.detach().clone()
    original_latent.requires_grad_(False)

    # float32 的可学习 latent 作为“攻击变量”
    latent_opt = torch.nn.Parameter(latent.detach().clone().to(device=device, dtype=torch.float32))
    latent_opt.requires_grad_(True)

    latent_lr = float(
        getattr(
            args,
            "lr",
            getattr(args, "latent_lr", 5e-4),
        )
    )
    optimizer = optim.AdamW([latent_opt], lr=latent_lr)
    cross_entro = torch.nn.CrossEntropyLoss()
    loss_l1 = torch.nn.L1Loss()
    lpips = LPIPS(net="vgg").cuda()
    lpips.requires_grad_(False)

    self_attn_weight = float(getattr(args, "self_attn_loss_weight", 0.0))
    cross_attn_weight = float(getattr(args, "cross_attn_loss_weight", 0.0))
    cross_attn_regularizer = None
    cross_attn_loss = torch.tensor(0.0, device=device)
    self_attn_regularizer = None
    self_attn_loss = torch.tensor(0.0, device=device)

    # 预处理后的图像也做一次 nan_to_num，避免后面参与 loss 出问题
    if freqsep_enabled:
        init_image = freqsep_base_diffusion
    else:
        init_image = preprocess(
            sr_operating_image, diffusion_res, device=device, dtype=vae_dtype
        )
        init_image = torch.nan_to_num(init_image, nan=0.0, posinf=1.0, neginf=-1.0)
        freqsep_base_compare = preprocess(
            sr_operating_image, res, device=device, dtype=vae_dtype
        )
        freqsep_base_compare = torch.nan_to_num(
            freqsep_base_compare, nan=0.0, posinf=1.0, neginf=-1.0
        )

    init_image_compare = preprocess(compare, res, device=device, dtype=vae_dtype)
    init_image_compare = torch.nan_to_num(
        init_image_compare, nan=0.0, posinf=1.0, neginf=-1.0
    )

    # mask 提前算好，后面循环内复用
    B, C, H, W = init_image.shape
    init_mask = torch.ones((B, 1, H, W), device=init_image.device, dtype=init_image.dtype)

    # ControlVAE 参数（供后面 is_encoder 分支用）
    params = {
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 512,
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
    }

    params_encoder = {
        "in_channels": params["in_channels"],
        "out_channels": params["latent_channels"],
        "down_block_types": params["down_block_types"],
        "block_out_channels": params["block_out_channels"],
        "layers_per_block": params["layers_per_block"],
        "act_fn": params["act_fn"],
        "norm_num_groups": params["norm_num_groups"],
        "double_z": True,
    }
    params_decoder = {
        "in_channels": params["latent_channels"],
        "out_channels": params["out_channels"],
        "up_block_types": params["up_block_types"],
        "block_out_channels": params["block_out_channels"],
        "layers_per_block": params["layers_per_block"],
        "act_fn": params["act_fn"],
        "norm_num_groups": params["norm_num_groups"],
    }

    if cross_attn_weight > 0:
        cross_attn_regularizer = CrossAttentionRegularizer.from_unet(
            model.unet,
            token_indices=prompt_token_indices,
            layers=str(getattr(args, "cross_attn_layers", "")),
            max_layers=int(getattr(args, "cross_attn_max_layers", 5)),
        )
        logger.info(
            f"[CROSS-ATTN] enabled weight={cross_attn_weight:.4f} layers={cross_attn_regularizer.layer_names} token_indices={prompt_token_indices}"
        )

    if self_attn_weight > 0:
        self_attn_regularizer = SelfAttentionRegularizer.from_unet(
            model.unet,
            layers=str(getattr(args, "self_attn_layers", "")),
            max_layers=int(getattr(args, "self_attn_max_layers", 5)),
            loss_type="mse",
        )
        logger.info(
            f"[SELF-ATTN] enabled weight={self_attn_weight:.4f} layers={self_attn_regularizer.layer_names}"
        )
        reset_scheduler_state(model)
        with torch.no_grad():
            self_attn_regularizer.begin_reference()
            ref_latents = torch.cat([original_latent, original_latent], dim=0)
            for ind, t in enumerate(active_timesteps):
                ref_latents = diffusion_step(model, ref_latents, context[ind], t, guidance_scale)
                ref_latents = torch.nan_to_num(ref_latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp_(-5.0, 5.0)
            self_attn_regularizer.end_reference()
        ref_step_counts = {name: len(vals) for name, vals in self_attn_regularizer.references.items()}
        logger.info(f"[SELF-ATTN] cached reference steps={ref_step_counts}")

    # ---- latent 迭代优化 ----
    pbar = tqdm(range(iterations), desc="Iterations")
    for _iter, _ in enumerate(pbar):
        if mode == "Generate":
            break
        reset_scheduler_state(model)

        # # 每一轮都重新作为 leaf，避免跨轮复用同一张图
        # latent_opt = latent_opt.detach().requires_grad_(True)
        # optimizer.param_groups[0]["params"] = [latent_opt]

        latents = torch.cat(
            [
                original_latent,                # fp16
                latent_opt.to(dtype=unet_dtype) # 转成 UNet 的 dtype
            ],
            dim=0,
        )

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            logger.warning(
                "latents has NaN/Inf BEFORE diffusion_step, applying nan_to_num."
            )
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
        latents = latents.clamp_(-5.0, 5.0)

        if self_attn_regularizer is not None:
            self_attn_regularizer.begin_compare()
        if cross_attn_regularizer is not None:
            cross_attn_regularizer.begin()

        # 扩散去噪
        for ind, t in enumerate(active_timesteps):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                logger.warning(
                    f"latents NaN/Inf at diffusion step {ind}, cleaning."
                )
                latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
            latents = latents.clamp_(-5.0, 5.0)

        if self_attn_regularizer is not None:
            self_attn_loss = self_attn_regularizer.end_compare(device=device)
        else:
            self_attn_loss = torch.tensor(0.0, device=device)
        if cross_attn_regularizer is not None:
            cross_attn_loss = cross_attn_regularizer.end(device=device)
        else:
            cross_attn_loss = torch.tensor(0.0, device=device)

        decode_latents = (latents / get_scaling_factor(model)).to(dtype=vae_dtype)

        if torch.isnan(decode_latents).any() or torch.isinf(decode_latents).any():
            logger.warning("decode_latents has NaN/Inf, cleaning.")
            decode_latents = torch.nan_to_num(
                decode_latents, nan=0.0, posinf=5.0, neginf=-5.0
            )
        decode_latents = decode_latents.clamp_(-5.0, 5.0)

        init_out_image = model.vae.decode(decode_latents)["sample"][1:] * init_mask + (
            1 - init_mask
        ) * init_image

        if torch.isnan(init_out_image).any() or torch.isinf(init_out_image).any():
            logger.warning("init_out_image has NaN/Inf, cleaning.")
            init_out_image = torch.nan_to_num(
                init_out_image, nan=0.0, posinf=1.0, neginf=-1.0
            )
        init_out_image = init_out_image.clamp_(-1.0, 1.0)

        metric_out_image = init_out_image
        if diffusion_res != res:
            metric_out_image = F.interpolate(
                init_out_image.float(),
                size=(res, res),
                mode="bilinear",
                align_corners=False,
            ).to(init_out_image.dtype)
        if freqsep_enabled and freqsep_recompose:
            init_out_image = recompose_with_generated_highfreq(
                init_out_image,
                init_image,
                scale=freqsep_scale,
            )
            metric_out_image = recompose_with_generated_highfreq(
                metric_out_image,
                freqsep_base_compare.to(dtype=metric_out_image.dtype),
                scale=freqsep_scale,
            )

        # 感知损失用 float32
        lpips_loss = lpips(init_image_compare.float(), metric_out_image.float()).mean()
        l1_loss_val = loss_l1(init_image_compare, metric_out_image)
        l1_loss_latent = loss_l1(original_latent.float(), latent_opt)

        out_image = (metric_out_image / 2 + 0.5).clamp(0, 1)

        # 这里不能 no_grad，要让分类器的梯度反传回 latent
        out_image_clf = normalize(out_image.float())
        # out_image_clf = normalize(out_image)
        pred_adv = classifier(out_image_clf) / 10.0
        attack_loss = -cross_entro(pred_adv, label) * args.attack_loss_weight

        # Paper uses coefficients alpha, beta, gamma. Here:
        # - lambda_l1   (alpha) for pixel L1
        # - lambda_lpips(beta)  for perceptual LPIPS
        # - attack_loss_weight (gamma) for attack objective
        lambda_l1 = float(getattr(args, "lambda_l1", 1.0))
        lambda_lpips = float(getattr(args, "lambda_lpips", 1.0))
        lambda_latent = float(getattr(args, "lambda_latent", 0.0))

        loss = (
            attack_loss
            + lambda_l1 * l1_loss_val
            + lambda_lpips * lpips_loss
            + lambda_latent * l1_loss_latent
            + cross_attn_weight * cross_attn_loss
            + self_attn_weight * self_attn_loss
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("loss is NaN/Inf, skip this optimization step.")
            continue

        if mode is None:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_([latent_opt], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                latent_opt.clamp_(-3.0, 3.0)

        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"l1_loss: {l1_loss_val.item():.5f} "
                f"l1_loss_latent: {l1_loss_latent.item():.5f} "
                f"lpips_loss: {lpips_loss.item():.5f} "
                f"cross_attn_loss: {float(cross_attn_loss.item()):.5f} "
                f"self_attn_loss: {float(self_attn_loss.item()):.5f} "
                f"loss: {loss.item():.5f}"
            )

    with torch.no_grad():
        reset_scheduler_state(model)
        inversion_latents_for_decode = torch.cat(
            [
                original_latent,
                original_latent,
            ],
            dim=0,
        )
        for ind, t in enumerate(active_timesteps):
            inversion_latents_for_decode = diffusion_step(
                model, inversion_latents_for_decode, context[ind], t, guidance_scale
            )
            inversion_latents_for_decode = torch.nan_to_num(
                inversion_latents_for_decode, nan=0.0, posinf=5.0, neginf=-5.0
            ).clamp_(-5.0, 5.0)
        inversion_only_image, inversion_only_eval = decode_latent_batch(
            model,
            inversion_latents_for_decode,
            init_image,
            init_mask,
            diffusion_res,
            res,
            vae_dtype,
        )
        if intermediate_dir_path is not None:
            save_uint8_image(
                tensor_image_to_uint8(inversion_only_image),
                intermediate_dir_path / "inversion_decode.png",
            )
            save_uint8_image(
                tensor_image_to_uint8(inversion_only_image),
                intermediate_dir_path / "entry_decode.png",
            )

    # ---- 最后再 decode 一次 ----
    with torch.no_grad():
        reset_scheduler_state(model)
        latents = torch.cat(
            [
                original_latent,
                latent_opt.to(dtype=unet_dtype),
            ],
            dim=0,
        )
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            logger.warning(
                "final latents has NaN/Inf BEFORE last diffusion, cleaning."
            )
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
        latents = latents.clamp_(-5.0, 5.0)

        for ind, t in enumerate(active_timesteps):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                logger.warning(
                    f"final latents NaN/Inf at diffusion step {ind}, cleaning."
                )
                latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
            latents = latents.clamp_(-5.0, 5.0)

    if self_attn_regularizer is not None:
        logger.info(
            f"[SELF-ATTN] final layers={self_attn_regularizer.layer_names} weight={self_attn_weight:.4f}"
        )
    if cross_attn_regularizer is not None:
        logger.info(
            f"[CROSS-ATTN] final layers={cross_attn_regularizer.layer_names} weight={cross_attn_weight:.4f} token_indices={prompt_token_indices} calls={cross_attn_regularizer.call_counts}"
        )

    # ====== decode 成图像 ======
    if args.is_encoder == 1:
        # ControlVAE 分支
        attack = torchattacks.PGD(
            classifier,
            eps=pgd_eps,
            alpha=float(pgd_alpha),
            steps=pgd_steps,
            random_start=False,
        )
        attack.set_normalization_used(mean=mean, std=std)
        normalize2 = transforms.Normalize(mean=mean, std=std)
        denormalize2 = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std],
        )
        control_attack_image = init_image
        if diffusion_res != res:
            control_attack_image = F.interpolate(
                init_image.float(),
                size=(res, res),
                mode="bilinear",
                align_corners=False,
            ).to(init_image.dtype)
        adv_images = attack(
            normalize2(control_attack_image * 0.5 + 0.5),
            label.detach().to(init_image.device),
        )
        if adm is not None:
            x_diff = adm.sdedit(denormalize2(adv_images), 1).detach()
            x_diff = torch.clamp(x_diff, 0, 1)
            adv_images = normalize2(x_diff)
        control_image = denormalize2(adv_images)
        if diffusion_res != res:
            control_image = F.interpolate(
                control_image.float(),
                size=(diffusion_res, diffusion_res),
                mode="bilinear",
                align_corners=False,
            ).to(adv_images.dtype)
        init_image1 = 2.0 * control_image - 1.0

        control_dtype = torch.float32
        init_image1 = init_image1.to(device=device, dtype=control_dtype)

        newencoder = NewEncoder(**params_encoder)
        p = args.encoder_weights
        state_dict = torch.load(p, map_location=device)
        newencoder.load_state_dict(state_dict["encoder"], strict=False)
        newencoder.requires_grad_(False)
        newencoder = newencoder.to(device=device, dtype=control_dtype)

        decoder = NewDecoder(**params_decoder)
        decoder.load_state_dict(model.vae.decoder.state_dict(), strict=False)
        decoder.requires_grad_(False)
        decoder = decoder.to(device=device, dtype=control_dtype)

        down_features = newencoder(init_image1)[1]

        decode_latents = model.vae.post_quant_conv(
            (latents / get_scaling_factor(model)).to(dtype=vae_dtype)
        )
        decode_latents = torch.nan_to_num(
            decode_latents, nan=0.0, posinf=5.0, neginf=-5.0
        )
        decode_latents = decode_latents.to(device=device, dtype=control_dtype)

        out_image = decoder(decode_latents, down_features)[1:]
        out_image = (out_image / 2 + 0.5).clamp(0, 1)
        out_image_eval = out_image
        if diffusion_res != res:
            out_image_eval = F.interpolate(
                out_image.float(),
                size=(res, res),
                mode="bilinear",
                align_corners=False,
            ).to(out_image.dtype)
    else:
        decoder = None
        down_features = None
        out_image, out_image_eval = decode_latent_batch(
            model,
            latents,
            init_image,
            init_mask,
            diffusion_res,
            res,
            vae_dtype,
        )
        if freqsep_enabled and freqsep_recompose:
            out_image = recompose_with_generated_highfreq(
                out_image,
                init_image,
                scale=freqsep_scale,
            )
            out_image_eval = recompose_with_generated_highfreq(
                out_image_eval,
                freqsep_base_compare.to(dtype=out_image_eval.dtype),
                scale=freqsep_scale,
            )

    normalize3 = transforms.Normalize(mean=mean, std=std)
    denormalize3 = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )

    out_image_norm = normalize3(out_image_eval.float())

    pred = classifier(out_image_norm)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (pred_label == label).sum().item() / len(label)
    logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    with torch.no_grad():
        pred1 = classifier_supp(out_image_norm)
        pred_label1 = torch.argmax(pred1, 1).detach()
        pred_accuracy1 = (pred_label1 == label).sum().item() / len(label)
        logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy1 * 100))

        pred2 = classifier_supp1(out_image_norm)
        pred_label2 = torch.argmax(pred2, 1).detach()
        pred_accuracy2 = (pred_label2 == label).sum().item() / len(label)
        logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy2 * 100))

        pred3 = classifier_supp2(out_image_norm)
        pred_label3 = torch.argmax(pred3, 1).detach()
        pred_accuracy3 = (pred_label3 == label).sum().item() / len(label)
        logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy3 * 100))

    logit = torch.nn.Softmax(dim=1)(pred)
    logit1 = torch.nn.Softmax(dim=1)(pred1)
    logit2 = torch.nn.Softmax(dim=1)(pred2)
    logit3 = torch.nn.Softmax(dim=1)(pred3)
    logger.info(f"after_pred: {pred_label}, {logit[0, pred_label[0]]}")
    logger.info(f"after_pred_supp: {pred_label1}, {logit1[0, pred_label1[0]]}")
    logger.info(f"after_pred_supp1: {pred_label2}, {logit2[0, pred_label2[0]]}")
    logger.info(f"after_pred_supp2: {pred_label3}, {logit3[0, pred_label3[0]]}")
    logger.info(f"after_true: {label}, {logit[0, label[0]]}")

    # ==========================================
    # ============= Visualization ==============
    # ==========================================

    save_image = torch.nan_to_num(out_image, nan=0.0, posinf=1.0, neginf=0.0)
    eval_image = denormalize3(out_image_norm)
    eval_image = torch.nan_to_num(eval_image, nan=0.0, posinf=1.0, neginf=0.0)

    imgg = np.clip(
        save_image.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0, 0, 255
    ).astype(np.uint8)

    # Save adversarial image.
    # Legacy behavior: save to f"{save_path}_adv_image.png" (save_path is a prefix).
    # New behavior: if out_path_adv is provided, save to that exact path.
    out_p = None
    if out_path_adv:
        out_p = Path(out_path_adv)
    elif save_path:
        out_p = Path(save_path + "_adv_image.png")
    if out_p is not None:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(imgg).save(str(out_p))
    if intermediate_dir_path is not None:
        save_uint8_image(imgg, intermediate_dir_path / "final_attack.png")

    real = (init_image_compare / 2 + 0.5).clamp(0, 1)
    real_np = (
        real[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
    ).astype(np.uint8)

    eval_np = np.clip(
        eval_image.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0, 0, 255
    ).astype(np.uint8)

    psnrv = calculate_psnr(eval_np, real_np, border=0)
    ssimv = calculate_ssim(eval_np, real_np, border=0)

    logger.info(f"psnrv: {psnrv}")
    logger.info(f"ssimv: {ssimv}")

    if self_attn_regularizer is not None:
        self_attn_regularizer.close()
    if cross_attn_regularizer is not None:
        cross_attn_regularizer.close()

    # Return accuracies (not class indices). `main.py` aggregates these across images.
    return (
        Image.fromarray(imgg),
        float(pred_accuracy),
        float(pred_accuracy1),
        float(pred_accuracy2),
        float(pred_accuracy3),
        psnrv,
        ssimv,
    )
