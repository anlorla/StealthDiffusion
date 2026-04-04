from __future__ import annotations

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import cv2
import math
import json
from lpips import LPIPS
from diffusers import EulerDiscreteScheduler
from diffusers.models.autoencoder_kl import (
    AutoencoderKL,
    Encoder,
    Decoder,
    AutoencoderKLOutput,
    DiagonalGaussianDistribution,
    DecoderOutput,
)
from ControlVAE import NewEncoder, NewDecoder, NewAutoencoderKL
from models.network_dncnn import DnCNN as DnCNNNet
import torchattacks
from cross_attn_regularizer import CrossAttentionRegularizer
from self_attn_regularizer import SelfAttentionRegularizer


_DNCNN_RESIDUAL_CACHE: dict[str, torch.nn.Module] = {}
_LPIPS_CACHE: dict[str, torch.nn.Module] = {}


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


def neg1pos1_to_01(image_tensor):
    return ((image_tensor.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def rgb01_to_gray(image_tensor):
    return (
        0.2989 * image_tensor[:, 0, :, :]
        + 0.5870 * image_tensor[:, 1, :, :]
        + 0.1140 * image_tensor[:, 2, :, :]
    )


def spectrum_log_amplitude_torch(gray_01):
    freq = torch.fft.fft2(gray_01.float(), dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    return torch.log1p(torch.abs(freq))


def load_dncnn_residual_extractor(device):
    device = torch.device(device)
    cache_key = str(device)
    cached = _DNCNN_RESIDUAL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    checkpoint_path = Path("/root/gpufree-data/checkpoints/dncnn_color_blind.pth")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing DnCNN checkpoint: {checkpoint_path}")

    state = torch.load(str(checkpoint_path), map_location="cpu")
    first_weight = next(v for k, v in state.items() if k.endswith(".weight") and v.ndim == 4)
    last_weight = next(v for k, v in reversed(list(state.items())) if k.endswith(".weight") and v.ndim == 4)
    num_convs = sum(1 for k, v in state.items() if k.endswith(".weight") and v.ndim == 4)
    act_mode = "BR" if any(k.endswith("running_mean") for k in state) else "R"

    model = DnCNNNet(
        in_nc=int(first_weight.shape[1]),
        out_nc=int(last_weight.shape[0]),
        nc=int(first_weight.shape[0]),
        nb=int(num_convs),
        act_mode=act_mode,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    _DNCNN_RESIDUAL_CACHE[cache_key] = model
    return model


def load_lpips_metric(device):
    device = torch.device(device)
    cache_key = str(device)
    cached = _LPIPS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = LPIPS(net="vgg").to(device)
    model.requires_grad_(False)
    model.eval()
    _LPIPS_CACHE[cache_key] = model
    return model


def compute_frequency_alignment_loss(pred_image, target_image, residual_extractor):
    pred_01 = neg1pos1_to_01(pred_image)
    target_01 = neg1pos1_to_01(target_image)

    pred_spec = spectrum_log_amplitude_torch(rgb01_to_gray(pred_01))
    target_spec = spectrum_log_amplitude_torch(rgb01_to_gray(target_01))
    img_freq_loss = torch.nn.functional.mse_loss(pred_spec, target_spec)

    if residual_extractor is None:
        zero = torch.zeros((), device=pred_image.device, dtype=img_freq_loss.dtype)
        return img_freq_loss, img_freq_loss, zero

    pred_residual = residual_extractor(pred_01)
    target_residual = residual_extractor(target_01)
    pred_residual_spec = spectrum_log_amplitude_torch(rgb01_to_gray(pred_residual))
    target_residual_spec = spectrum_log_amplitude_torch(rgb01_to_gray(target_residual))
    residual_freq_loss = torch.nn.functional.mse_loss(pred_residual_spec, target_residual_spec)
    total_freq_loss = 0.5 * (img_freq_loss + residual_freq_loss)
    return total_freq_loss, img_freq_loss, residual_freq_loss


def is_pixart_alpha_backbone(model):
    return model.__class__.__name__ == "PixArtAlphaPipeline" or hasattr(model, "transformer")


def get_denoiser(model):
    return model.transformer if is_pixart_alpha_backbone(model) else model.unet


def get_scaling_factor(model):
    return float(getattr(model.vae.config, "scaling_factor", 0.18215) or 0.18215)


def move_pipeline_modules_to_cpu(model, module_names):
    moved = []
    for name in module_names:
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.to("cpu")
            moved.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def move_pipeline_modules_to_device(model, module_names, device):
    moved = []
    for name in module_names:
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)
            moved.append(name)
    return moved


def get_pipeline_execution_device(model):
    device = getattr(model, "_execution_device", None)
    if device is None:
        device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        device = next(get_denoiser(model).parameters()).device
    return torch.device(device)


def ensure_core_pipeline_modules_on_device(model):
    device = torch.device("cuda") if torch.cuda.is_available() else get_pipeline_execution_device(model)
    module_names = ["vae"]
    module_names.append("transformer" if is_pixart_alpha_backbone(model) else "unet")
    move_pipeline_modules_to_device(model, module_names, device)
    return device


def ensure_scheduler_tensors_on_device(scheduler, device):
    device = torch.device(device)
    for name in (
        "timesteps",
        "sigmas",
        "alphas_cumprod",
        "final_alpha_cumprod",
        "alphas",
        "betas",
    ):
        value = getattr(scheduler, name, None)
        if torch.is_tensor(value) and value.device != device:
            setattr(scheduler, name, value.to(device))
    return scheduler


def offload_unused_conditioning_modules(model):
    return move_pipeline_modules_to_cpu(
        model,
        [
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
            "safety_checker",
        ],
    )


def encode_prompt_embeddings(model, prompt):
    prompt = prompt if isinstance(prompt, list) else [prompt]
    exec_device = ensure_core_pipeline_modules_on_device(model)

    if is_pixart_alpha_backbone(model):
        move_pipeline_modules_to_device(model, ["text_encoder"], exec_device)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = model.encode_prompt(
            prompt,
            do_classifier_free_guidance=True,
            negative_prompt="",
            num_images_per_prompt=1,
            device=exec_device,
            clean_caption=False,
        )
        return {
            "text_embeddings": prompt_embeds,
            "text_attention_mask": prompt_attention_mask,
            "uncond_embeddings": negative_prompt_embeds,
            "uncond_attention_mask": negative_prompt_attention_mask,
        }

    batch_size = len(prompt)
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(exec_device))[0]

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(exec_device))[0]
    return {
        "text_embeddings": text_embeddings,
        "uncond_embeddings": uncond_embeddings,
    }


def build_added_cond_kwargs(model, height, width, batch_size, dtype, device):
    if not is_pixart_alpha_backbone(model):
        return None

    sample_size = getattr(model.transformer.config, "sample_size", None)
    if sample_size != 128:
        return {"resolution": None, "aspect_ratio": None}

    resolution = torch.tensor([height, width]).repeat(batch_size, 1).to(dtype=dtype, device=device)
    aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1).to(dtype=dtype, device=device)
    return {"resolution": resolution, "aspect_ratio": aspect_ratio}


def build_context(
    model,
    prompt,
    res,
    uncond_embeddings=None,
    uncond_attention_mask=None,
):
    prompt = prompt if isinstance(prompt, list) else [prompt]
    prompt_info = encode_prompt_embeddings(model, prompt)
    text_embeddings = prompt_info["text_embeddings"]

    if is_pixart_alpha_backbone(model):
        if uncond_embeddings is None:
            uncond_embeddings = prompt_info["uncond_embeddings"]
        if uncond_attention_mask is None:
            uncond_attention_mask = prompt_info["uncond_attention_mask"]

        batch_size = text_embeddings.shape[0]
        return {
            "prompt_embeds": torch.cat([uncond_embeddings, text_embeddings], dim=0),
            "prompt_attention_mask": torch.cat([uncond_attention_mask, prompt_info["text_attention_mask"]], dim=0),
            "added_cond_kwargs": build_added_cond_kwargs(
                model,
                height=res,
                width=res,
                batch_size=batch_size,
                dtype=text_embeddings.dtype,
                device=text_embeddings.device,
            ),
        }

    if uncond_embeddings is None:
        uncond_embeddings = prompt_info["uncond_embeddings"]
    return {
        "encoder_hidden_states": torch.cat([uncond_embeddings, text_embeddings], dim=0),
    }


def resolve_prompt_text_for_attack(imagenet_label, label_value: int, prompt_text):
    base_prompt = (prompt_text or imagenet_label.refined_Label[label_value]).strip()
    if not base_prompt:
        raise ValueError('Resolved prompt_text is empty')
    return base_prompt


def resolve_prompt_token_indices(tokenizer, prompt_text: str):
    max_length = getattr(tokenizer, 'model_max_length', 120)
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
    ensure_core_pipeline_modules_on_device(model)
    denoiser_param = next(get_denoiser(model).parameters())
    ensure_scheduler_tensors_on_device(model.scheduler, denoiser_param.device)
    latents = latents.to(device=denoiser_param.device, dtype=denoiser_param.dtype)
    latent_model_input = torch.cat([latents] * 2)

    if is_pixart_alpha_backbone(model):
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], dtype=torch.int64, device=latent_model_input.device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(latent_model_input.device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])

        noise_pred = model.transformer(
            latent_model_input,
            encoder_hidden_states=context["prompt_embeds"],
            encoder_attention_mask=context["prompt_attention_mask"],
            timestep=current_timestep,
            added_cond_kwargs=context["added_cond_kwargs"],
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latent_channels = latents.shape[1]
        if model.transformer.config.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        return noise_pred

    noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=context["encoder_hidden_states"])["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)


def encoder(image, model, res=512, sample_posterior=True):
    """
    用 VAE 的 device/dtype 做预处理，再把 latent 转成 denoiser 的 dtype
    """
    ensure_core_pipeline_modules_on_device(model)
    vae_param = next(model.vae.parameters())
    denoiser_param = next(get_denoiser(model).parameters())
    vae_device = vae_param.device
    vae_dtype = vae_param.dtype
    denoiser_dtype = denoiser_param.dtype

    image = preprocess(image, res, device=vae_device, dtype=vae_dtype)
    posterior = model.vae.encode(image).latent_dist
    if sample_posterior:
        generator = torch.Generator(device=vae_device).manual_seed(8888)
        latents = get_scaling_factor(model) * posterior.sample(generator=generator)
    else:
        latents = get_scaling_factor(model) * posterior.mode()

    latents = latents.to(dtype=denoiser_dtype)
    return latents

# def encoder(image, model, res=512):
#     device = next(model.vae.parameters()).device

#     generator = torch.Generator(device=device).manual_seed(8888)
#     image = preprocess(image, res, device=device, dtype=torch.float32)

#     gpu_generator = torch.Generator(device=device)
#     gpu_generator.manual_seed(generator.initial_seed())

#     latents = 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)
#     # ★ 保持 float32，不要再 cast 到 half
#     return latents


@torch.no_grad()
def ddim_reverse_sample(
    image,
    prompt,
    model,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    res=512,
    sample_posterior=True,
):
    """
    ==========================================
    ============ DDIM Inversion ==============
    ==========================================
    """
    ensure_core_pipeline_modules_on_device(model)
    if is_pixart_alpha_backbone(model) and getattr(model, "inverse_scheduler", None) is not None:
        exec_device = get_pipeline_execution_device(model)
        model.scheduler.set_timesteps(num_inference_steps, device=exec_device)
        model.inverse_scheduler.set_timesteps(num_inference_steps, device=exec_device)

        latents = encoder(image, model, res=res, sample_posterior=sample_posterior)
        timesteps = model.inverse_scheduler.timesteps
        context = build_context(model, [prompt[0]], res=res)

        all_latents = [latents]
        for t in tqdm(timesteps[1:], desc="DDIM_inverse"):
            model_output = get_model_prediction(model, latents, context, t, guidance_scale)
            latents = model.inverse_scheduler.step(model_output, t, latents)["prev_sample"]
            all_latents.append(latents)
        return latents, all_latents

    context = build_context(model, [prompt[0]], res=res)
    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res, sample_posterior=sample_posterior)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        noise_pred = get_model_prediction(model, latents, context, t, guidance_scale)

        next_timestep = (
            t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        )
        alpha_bar_next = (
            model.scheduler.alphas_cumprod[next_timestep]
            if next_timestep <= model.scheduler.config.num_train_timesteps
            else torch.tensor(0.0)
        )

        reverse_x0 = (
            1 / torch.sqrt(model.scheduler.alphas_cumprod[t])
            * (latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t]))
        )

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(
            1 - alpha_bar_next
        ) * noise_pred

        all_latents.append(latents)

    # all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


def build_start_latent(
    image,
    prompt,
    model,
    num_inference_steps: int,
    res: int,
    start_step: int,
    startpoint_mode: str,
):
    ensure_core_pipeline_modules_on_device(model)
    exec_device = get_pipeline_execution_device(model)
    model.scheduler.set_timesteps(num_inference_steps, device=exec_device)
    timesteps = model.scheduler.timesteps

    if startpoint_mode == "ddim_inversion":
        latent, inversion_latents = ddim_reverse_sample(
            image,
            prompt,
            model,
            num_inference_steps,
            0,
            res=res,
        )
        inversion_latents = inversion_latents[::-1]
        chosen_index = max(0, min(int(start_step) - 1, len(inversion_latents) - 1))
        latent = inversion_latents[chosen_index]
        active_timesteps = timesteps[chosen_index + 1 :]
        chosen_timestep = active_timesteps[0] if len(active_timesteps) > 0 else torch.tensor(0, device=timesteps.device, dtype=timesteps.dtype)
        return {
            "latent": latent,
            "active_timesteps": active_timesteps,
            "requested_start_step": int(start_step),
            "chosen_index": int(chosen_index),
            "chosen_timestep": chosen_timestep,
            "startpoint_family": "ddim",
            "startpoint_value": f"t_start_{int(start_step)}",
        }

    z0 = encoder(image, model, res=res, sample_posterior=True)
    z0 = z0.to(device=exec_device, dtype=next(get_denoiser(model).parameters()).dtype)

    if startpoint_mode == "clean_latent":
        active_timesteps = timesteps[:0]
        return {
            "latent": z0,
            "active_timesteps": active_timesteps,
            "requested_start_step": int(start_step),
            "chosen_index": int(len(timesteps)),
            "chosen_timestep": torch.tensor(0, device=timesteps.device, dtype=timesteps.dtype),
            "startpoint_family": "clean",
            "startpoint_value": "z0",
        }

    if startpoint_mode == "euler_add_noise":
        euler_scheduler = EulerDiscreteScheduler.from_config(dict(model.scheduler.config))
        euler_scheduler.set_timesteps(num_inference_steps, device=exec_device)
        euler_timesteps = euler_scheduler.timesteps
        sigma_index = max(0, min(int(start_step), len(euler_timesteps) - 1))
        chosen_t = euler_timesteps[sigma_index]
        active_timesteps = timesteps[sigma_index:]
        noise = torch.randn_like(z0)
        zt = euler_scheduler.add_noise(z0, noise, chosen_t.unsqueeze(0))
        zt = zt.to(device=exec_device, dtype=next(get_denoiser(model).parameters()).dtype)
        sigma_value = float(euler_scheduler.sigmas[sigma_index].item()) if hasattr(euler_scheduler, "sigmas") else float("nan")
        return {
            "latent": zt,
            "active_timesteps": active_timesteps,
            "requested_start_step": int(start_step),
            "chosen_index": int(sigma_index),
            "chosen_timestep": chosen_t,
            "sigma_value": sigma_value,
            "startpoint_family": "euler",
            "startpoint_value": f"sigma_{sigma_index}",
        }

    raise ValueError(f"Unsupported startpoint_mode={startpoint_mode!r}")


def init_latent(latent, model, height, width, batch_size):
    denoiser = get_denoiser(model)
    latents = latent.expand(
        batch_size, denoiser.config.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    denoiser_param = next(get_denoiser(model).parameters())
    ensure_scheduler_tensors_on_device(model.scheduler, denoiser_param.device)
    latents = latents.to(device=denoiser_param.device, dtype=denoiser_param.dtype)
    noise_pred = get_model_prediction(model, latents, context, t, guidance_scale)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = latents.to(device=denoiser_param.device, dtype=denoiser_param.dtype)
    return latents


def latent2image(vae, latents, args, decoder, down_features):
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215) or 0.18215)
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
    prompt_text=None,
):
    if args.dataset_name == "ours_try":
        from dataset_caption import ours_label as imagenet_label
    else:
        raise NotImplementedError

    # VAE / denoiser 精度 & device
    ensure_core_pipeline_modules_on_device(model)
    denoiser_param = next(get_denoiser(model).parameters())
    vae_param = next(model.vae.parameters())
    device = denoiser_param.device
    denoiser_dtype = denoiser_param.dtype
    vae_dtype = vae_param.dtype

    label = torch.from_numpy(label).long().cuda()

    classifier, classifier_supp, classifier_supp1, classifier_supp2 = classes
    classifier = classifier.eval()
    classifier_supp = classifier_supp.eval()
    classifier_supp1 = classifier_supp1.eval()
    classifier_supp2 = classifier_supp2.eval()
    ordered_names = [s.strip() for s in str(getattr(args, "model_name", "E,R,D,S")).split(",") if s.strip()]
    if len(ordered_names) != 4:
        raise ValueError(f"Expected 4 detector names in args.model_name, got {ordered_names}")
    classifier_map = {
        ordered_names[0]: classifier,
        ordered_names[1]: classifier_supp,
        ordered_names[2]: classifier_supp1,
        ordered_names[3]: classifier_supp2,
    }
    attack_surrogates_raw = str(getattr(args, "attack_surrogates", "")).strip()
    if attack_surrogates_raw:
        attack_surrogates = [s.strip() for s in attack_surrogates_raw.split(",") if s.strip()]
    else:
        attack_surrogates = [ordered_names[0]]
    if len(set(attack_surrogates)) != len(attack_surrogates):
        raise ValueError(f"Duplicate names in attack_surrogates={attack_surrogates}")
    unknown_attack = [s for s in attack_surrogates if s not in classifier_map]
    if unknown_attack:
        raise ValueError(
            f"attack_surrogates contains names outside model_name order: {unknown_attack} vs {ordered_names}"
        )

    height = width = res

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

    # Use the true label (0=fake, 1=real) for the PGD warm start; hard-coding 0
    # breaks experiments when running on real images.
    adv_images = attack(test_image, label.detach().to(test_image.device))
    imgg = np.uint8(
        denormalize(adv_images).detach().cpu().numpy()[0].transpose((1, 2, 0)) * 255.0
    )
    image = Image.fromarray(imgg)

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

    tokenizer = getattr(model, "tokenizer", None)
    prompt_token_indices = resolve_prompt_token_indices(tokenizer, base_prompt) if tokenizer is not None else []
    if getattr(args, 'cross_attn_loss_weight', 0.0) > 0 and not prompt_token_indices:
        raise ValueError(f"No valid prompt token indices resolved for prompt_text={base_prompt!r}")
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        decoder_tokens = tokenizer.encode(base_prompt)
        logger.info(f"decoder: {decoder_tokens}")
    else:
        logger.info("decoder token ids unavailable for current backbone")
    logger.info(f"[PROMPT] base_prompt={base_prompt!r} token_indices={prompt_token_indices}")

    startpoint_mode = str(getattr(args, "startpoint_mode", "ddim_inversion"))
    start_state = build_start_latent(
        image=image,
        prompt=prompt,
        model=model,
        num_inference_steps=num_inference_steps,
        res=height,
        start_step=start_step,
        startpoint_mode=startpoint_mode,
    )
    latent = start_state["latent"]
    active_timesteps = start_state["active_timesteps"]
    chosen_timestep = start_state["chosen_timestep"]
    logger.info(
        "[STARTPOINT] mode=%s family=%s requested_start_step=%s chosen_index=%s chosen_timestep=%s startpoint_value=%s",
        startpoint_mode,
        start_state.get("startpoint_family", ""),
        start_state.get("requested_start_step", ""),
        start_state.get("chosen_index", ""),
        float(chosen_timestep.item()) if torch.is_tensor(chosen_timestep) else chosen_timestep,
        start_state.get("startpoint_value", ""),
    )
    if "sigma_value" in start_state:
        logger.info("[STARTPOINT] euler_sigma=%.6f", float(start_state["sigma_value"]))
    logger.info(
        "[STARTPOINT] latent mean=%.4f std=%.4f nan=%s inf=%s active_steps=%d",
        latent.mean().item(),
        latent.std().item(),
        bool(torch.isnan(latent).any().item()),
        bool(torch.isinf(latent).any().item()),
        int(len(active_timesteps)),
    )
    if getattr(args, "save_intermediates", False) and getattr(args, "intermediate_root", ""):
        meta_path = Path(args.intermediate_root) / "startpoint_meta.json"
        meta = {
            "startpoint_mode": startpoint_mode,
            "startpoint_family": start_state.get("startpoint_family", ""),
            "startpoint_value": start_state.get("startpoint_value", ""),
            "requested_start_step": int(start_state.get("requested_start_step", start_step)),
            "chosen_index": int(start_state.get("chosen_index", 0)),
            "chosen_timestep": float(chosen_timestep.item()) if torch.is_tensor(chosen_timestep) else chosen_timestep,
            "active_step_count": int(len(active_timesteps)),
            "latent_mean": float(latent.mean().item()),
            "latent_std": float(latent.std().item()),
        }
        if "sigma_value" in start_state:
            meta["sigma_value"] = float(start_state["sigma_value"])
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    init_prompt = [base_prompt]

    batch_size_init = len(init_prompt)
    with torch.no_grad():
        init_prompt_info = encode_prompt_embeddings(model, init_prompt)
    uncond_embeddings = init_prompt_info["uncond_embeddings"]
    uncond_attention_mask = init_prompt_info.get("uncond_attention_mask")

    all_uncond_emb = []
    all_uncond_masks = []
    latent, latents = init_latent(latent, model, height, width, batch_size_init)
    with torch.no_grad():
        for k, z in enumerate(latents):
            print(
                f"[INV] step {k}: mean={z.mean().item():.4f}, std={z.std().item():.4f}, "
                f"nan={torch.isnan(z).any().item()}, inf={torch.isinf(z).any().item()}"
            )

    for _ind, _t in enumerate(
        tqdm(active_timesteps, desc="Optimize_uncond_embed")
    ):
        all_uncond_emb.append(uncond_embeddings.detach().clone())
        if uncond_attention_mask is not None:
            all_uncond_masks.append(uncond_attention_mask.detach().clone())

    # ==========================================
    # ============ Latents Attack ==============
    # ==========================================

    batch_size = len(prompt)

    context = []
    for i in range(len(all_uncond_emb)):
        uncond_mask = None
        if all_uncond_masks:
            uncond_mask = all_uncond_masks[i].repeat(batch_size, 1)
        with torch.no_grad():
            context_i = build_context(
                model,
                prompt,
                res=height,
                uncond_embeddings=all_uncond_emb[i].repeat(batch_size, 1, 1),
                uncond_attention_mask=uncond_mask,
            )
        for key, value in context_i.items():
            if torch.is_tensor(value):
                context_i[key] = value.detach()
            elif isinstance(value, dict):
                context_i[key] = {
                    subkey: subvalue.detach() if torch.is_tensor(subvalue) else subvalue
                    for subkey, subvalue in value.items()
                }
        context.append(context_i)

    offload_unused_conditioning_modules(model)
    ensure_core_pipeline_modules_on_device(model)

    get_denoiser(model).requires_grad_(False)
    model.vae.requires_grad_(False)


    # inversion 得到的 latent（和 UNet 同 dtype，比如 float16），只用来做对比
    original_latent = latent.detach().clone()
    original_latent.requires_grad_(False)

    # float32 的可学习 latent 作为“攻击变量”
    latent_opt = torch.nn.Parameter(latent.detach().clone().to(device=device, dtype=torch.float32))
    latent_opt.requires_grad_(True)

    optimizer = optim.AdamW([latent_opt], lr=float(getattr(args, "lr", 1e-3)))
    cross_entro = torch.nn.CrossEntropyLoss()
    loss_l1 = torch.nn.L1Loss()
    lpips = load_lpips_metric(device)

    self_attn_weight = float(getattr(args, "self_attn_loss_weight", 0.0))
    cross_attn_weight = float(getattr(args, "cross_attn_loss_weight", 0.0))
    lambda_f = float(getattr(args, "lambda_f", 0.0))
    cross_attn_regularizer = None
    cross_attn_loss = torch.tensor(0.0, device=device)
    self_attn_regularizer = None
    self_attn_loss = torch.tensor(0.0, device=device)
    frequency_residual_extractor = None
    if lambda_f > 0:
        frequency_residual_extractor = load_dncnn_residual_extractor(device)
        logger.info(f"[FREQ-LOSS] enabled lambda_f={lambda_f:.4f}")

    # 预处理后的图像也做一次 nan_to_num，避免后面参与 loss 出问题
    init_image = preprocess(image, res, device=device, dtype=vae_dtype)
    init_image = torch.nan_to_num(init_image, nan=0.0, posinf=1.0, neginf=-1.0)

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
        cross_attn_regularizer = CrossAttentionRegularizer.from_pixart(
            get_denoiser(model),
            token_indices=prompt_token_indices,
            layers=str(getattr(args, "cross_attn_layers", "")),
        )
        logger.info(
            f"[CROSS-ATTN] enabled weight={cross_attn_weight:.4f} layers={cross_attn_regularizer.layer_names} token_indices={prompt_token_indices}"
        )

    if self_attn_weight > 0:
        self_attn_regularizer = SelfAttentionRegularizer.from_pixart(
            get_denoiser(model),
            layers=str(getattr(args, "self_attn_layers", "")),
            loss_type="fro",
        )
        logger.info(
            f"[SELF-ATTN] enabled weight={self_attn_weight:.4f} layers={self_attn_regularizer.layer_names}"
        )
        with torch.no_grad():
            self_attn_regularizer.begin_reference()
            ref_latents = torch.cat([original_latent, original_latent], dim=0)
            for ind, t in enumerate(active_timesteps):
                ref_latents = diffusion_step(model, ref_latents, context[ind], t, guidance_scale)
                ref_latents = torch.nan_to_num(ref_latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp_(-5.0, 5.0)
            self_attn_regularizer.end_reference()
        ref_step_counts = {name: len(vals) for name, vals in self_attn_regularizer.references.items()}
        logger.info(f"[SELF-ATTN] cached reference steps={ref_step_counts}")

    # ---- DDIM 迭代优化 ----
    pbar = tqdm(range(iterations), desc="Iterations")
    for _iter, _ in enumerate(pbar):
        if mode == "Generate":
            break

        # # 每一轮都重新作为 leaf，避免跨轮复用同一张图
        # latent_opt = latent_opt.detach().requires_grad_(True)
        # optimizer.param_groups[0]["params"] = [latent_opt]

        latents = torch.cat(
            [
                original_latent.to(device=device, dtype=denoiser_dtype),
                latent_opt.to(device=device, dtype=denoiser_dtype),
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

        # DDIM 正向采样
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

        # Only decode the adversarial branch; the clean reference branch is unused here.
        decode_latents = (latents[1:] / get_scaling_factor(model)).to(dtype=vae_dtype)

        if torch.isnan(decode_latents).any() or torch.isinf(decode_latents).any():
            logger.warning("decode_latents has NaN/Inf, cleaning.")
            decode_latents = torch.nan_to_num(
                decode_latents, nan=0.0, posinf=5.0, neginf=-5.0
            )
        decode_latents = decode_latents.clamp_(-5.0, 5.0)

        init_out_image = model.vae.decode(decode_latents)["sample"] * init_mask + (
            1 - init_mask
        ) * init_image

        if torch.isnan(init_out_image).any() or torch.isinf(init_out_image).any():
            logger.warning("init_out_image has NaN/Inf, cleaning.")
            init_out_image = torch.nan_to_num(
                init_out_image, nan=0.0, posinf=1.0, neginf=-1.0
            )
        init_out_image = init_out_image.clamp_(-1.0, 1.0)

        # 感知损失用 float32
        lpips_loss = lpips(init_image_compare.float(), init_out_image.float()).mean()
        l1_loss_val = loss_l1(init_image_compare, init_out_image)
        l1_loss_latent = loss_l1(original_latent.float(), latent_opt)
        if lambda_f > 0:
            freq_loss, img_freq_loss, residual_freq_loss = compute_frequency_alignment_loss(
                init_out_image,
                init_image_compare,
                frequency_residual_extractor,
            )
        else:
            freq_loss = torch.tensor(0.0, device=device)
            img_freq_loss = torch.tensor(0.0, device=device)
            residual_freq_loss = torch.tensor(0.0, device=device)

        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)

        # 这里不能 no_grad，要让分类器的梯度反传回 latent
        out_image_clf = normalize(out_image.float())
        # out_image_clf = normalize(out_image)
        attack_loss_terms = []
        attack_loss_by_surrogate = {}
        pred_adv = None
        for surrogate_name in attack_surrogates:
            pred_adv_current = classifier_map[surrogate_name](out_image_clf) / 10.0
            if pred_adv is None and surrogate_name == ordered_names[0]:
                pred_adv = pred_adv_current
            attack_loss_current = -cross_entro(pred_adv_current, label) * args.attack_loss_weight
            attack_loss_terms.append(attack_loss_current)
            attack_loss_by_surrogate[surrogate_name] = float(attack_loss_current.detach().item())
        if pred_adv is None:
            pred_adv = classifier(out_image_clf) / 10.0
        attack_loss = torch.stack(attack_loss_terms).mean()

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
            + lambda_f * freq_loss
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
                f"freq_loss: {freq_loss.item():.5f} "
                f"img_freq_loss: {img_freq_loss.item():.5f} "
                f"residual_freq_loss: {residual_freq_loss.item():.5f} "
                f"cross_attn_loss: {float(cross_attn_loss.item()):.5f} "
                f"self_attn_loss: {float(self_attn_loss.item()):.5f} "
                f"loss: {loss.item():.5f}"
            )
            if len(attack_surrogates) > 1:
                logger.info(
                    f"[ENSEMBLE] attack_surrogates={attack_surrogates} "
                    f"component_attack_loss={attack_loss_by_surrogate}"
                )

    # ---- 最后再 decode 一次 ----
    with torch.no_grad():
        latents = torch.cat(
            [
                original_latent,
                latent_opt.to(dtype=denoiser_dtype),
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
        adv_images = attack(
            normalize2(init_image * 0.5 + 0.5), label.detach().to(init_image.device)
        )
        if adm is not None:
            x_diff = adm.sdedit(denormalize2(adv_images), 1).detach()
            x_diff = torch.clamp(x_diff, 0, 1)
            adv_images = normalize2(x_diff)
        init_image1 = 2.0 * denormalize2(adv_images) - 1.0

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
    else:
        decoder = None
        down_features = None

        decode_latents = (latents.detach()[1:] / get_scaling_factor(model)).to(dtype=vae_dtype)
        decode_latents = torch.nan_to_num(
            decode_latents, nan=0.0, posinf=5.0, neginf=-5.0
        )
        out_image = model.vae.decode(decode_latents)["sample"] * init_mask + (
            1 - init_mask
        ) * init_image

    out_image = (out_image / 2 + 0.5).clamp(0, 1)

    normalize3 = transforms.Normalize(mean=mean, std=std)
    denormalize3 = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )

    out_image_norm = normalize3(out_image.float())

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

    denorm_img = denormalize3(out_image_norm)
    denorm_img = torch.nan_to_num(denorm_img, nan=0.0, posinf=1.0, neginf=0.0)

    imgg = np.clip(
        denorm_img.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0, 0, 255
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

    real = (init_image_compare / 2 + 0.5).clamp(0, 1)
    real_np = (
        real[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
    ).astype(np.uint8)

    psnrv = calculate_psnr(imgg, real_np, border=0)
    ssimv = calculate_ssim(imgg, real_np, border=0)

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
