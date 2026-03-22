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
from lpips import LPIPS
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


def is_pixart_alpha_backbone(model):
    return model.__class__.__name__ == "PixArtAlphaPipeline" or hasattr(model, "transformer")


def get_denoiser(model):
    return model.transformer if is_pixart_alpha_backbone(model) else model.unet


def get_scaling_factor(model):
    return float(getattr(model.vae.config, "scaling_factor", 0.18215) or 0.18215)


def encode_prompt_embeddings(model, prompt):
    prompt = prompt if isinstance(prompt, list) else [prompt]

    if is_pixart_alpha_backbone(model):
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
            device=model.device,
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
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
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


def get_model_prediction(model, latents, context, t, guidance_scale):
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


def encoder(image, model, res=512):
    """
    用 VAE 的 device/dtype 做预处理，再把 latent 转成 denoiser 的 dtype
    """
    vae_param = next(model.vae.parameters())
    denoiser_param = next(get_denoiser(model).parameters())
    vae_device = vae_param.device
    vae_dtype = vae_param.dtype
    denoiser_dtype = denoiser_param.dtype

    generator = torch.Generator(device=vae_device).manual_seed(8888)
    image = preprocess(image, res, device=vae_device, dtype=vae_dtype)

    gpu_generator = torch.Generator(device=vae_device)
    gpu_generator.manual_seed(generator.initial_seed())

    latents = get_scaling_factor(model) * model.vae.encode(image).latent_dist.sample(
        generator=gpu_generator
    )

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
):
    """
    ==========================================
    ============ DDIM Inversion ==============
    ==========================================
    """
    if is_pixart_alpha_backbone(model) and getattr(model, "inverse_scheduler", None) is not None:
        model.scheduler.set_timesteps(num_inference_steps, device=model.device)
        model.inverse_scheduler.set_timesteps(num_inference_steps, device=model.device)

        latents = encoder(image, model, res=res)
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

    latents = encoder(image, model, res=res)
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


def init_latent(latent, model, height, width, batch_size):
    denoiser = get_denoiser(model)
    latents = latent.expand(
        batch_size, denoiser.config.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    noise_pred = get_model_prediction(model, latents, context, t, guidance_scale)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
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
):
    if args.dataset_name == "ours_try":
        from dataset_caption import ours_label as imagenet_label
    else:
        raise NotImplementedError

    # VAE / denoiser 精度 & device
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
    target_prompt = " ".join(
        [imagenet_label.refined_Label[label.item()] for _ in range(1, topN)]
    )

    prompt = [""] * 2
    logger.info(
        f"prompt generate: {prompt[0]} \tlabels: {pred_labels.cpu().numpy().tolist()}"
    )

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        true_label = tokenizer.encode(imagenet_label.refined_Label[label.item()])
        target_label = tokenizer.encode(target_prompt)
        logger.info(f"decoder: {true_label}, {target_label}")
    else:
        logger.info("decoder token ids unavailable for current backbone")

    # ==========================================
    # ============ DDIM Inversion ==============
    # ==========================================
    latent, inversion_latents = ddim_reverse_sample(
        image,
        prompt,
        model,
        num_inference_steps,
        0,
        res=height,
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

    init_prompt = [""]

    batch_size_init = len(init_prompt)
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

    uncond_embeddings.requires_grad_(True)

    for _ind, _t in enumerate(
        tqdm(model.scheduler.timesteps[1 + start_step - 1 :], desc="Optimize_uncond_embed")
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
        context.append(
            build_context(
                model,
                prompt,
                res=height,
                uncond_embeddings=all_uncond_emb[i].repeat(batch_size, 1, 1),
                uncond_attention_mask=uncond_mask,
            )
        )

    get_denoiser(model).requires_grad_(False)
    model.vae.requires_grad_(False)


    # inversion 得到的 latent（和 UNet 同 dtype，比如 float16），只用来做对比
    original_latent = latent.detach().clone()
    original_latent.requires_grad_(False)

    # float32 的可学习 latent 作为“攻击变量”
    latent_opt = torch.nn.Parameter(latent.detach().clone().to(device=device, dtype=torch.float32))
    latent_opt.requires_grad_(True)

    optimizer = optim.AdamW([latent_opt], lr=5e-4)
    cross_entro = torch.nn.CrossEntropyLoss()
    loss_l1 = torch.nn.L1Loss()
    lpips = LPIPS(net="vgg").cuda()
    lpips.requires_grad_(False)

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
                original_latent,
                latent_opt.to(dtype=denoiser_dtype)
            ],
            dim=0,
        )

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            logger.warning(
                "latents has NaN/Inf BEFORE diffusion_step, applying nan_to_num."
            )
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
        latents = latents.clamp_(-5.0, 5.0)

        # DDIM 正向采样
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1 :]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                logger.warning(
                    f"latents NaN/Inf at diffusion step {ind}, cleaning."
                )
                latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
            latents = latents.clamp_(-5.0, 5.0)

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

        # 感知损失用 float32
        lpips_loss = lpips(init_image_compare.float(), init_out_image.float()).mean()
        l1_loss_val = loss_l1(init_image_compare, init_out_image)
        l1_loss_latent = loss_l1(original_latent.float(), latent_opt)

        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)

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
                f"loss: {loss.item():.5f}"
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

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1 :]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                logger.warning(
                    f"final latents NaN/Inf at diffusion step {ind}, cleaning."
                )
                latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
            latents = latents.clamp_(-5.0, 5.0)

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

        decode_latents = (latents.detach() / get_scaling_factor(model)).to(dtype=vae_dtype)
        decode_latents = torch.nan_to_num(
            decode_latents, nan=0.0, posinf=5.0, neginf=-5.0
        )
        out_image = model.vae.decode(decode_latents)["sample"][1:] * init_mask + (
            1 - init_mask
        ) * init_image

    out_image = (out_image / 2 + 0.5).clamp(0, 1)

    normalize3 = transforms.Normalize(mean=mean, std=std)
    denormalize3 = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )

    out_image_norm = normalize3(out_image)

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
