#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_ROOT = Path('/root/gpufree-data')
PIXART_ROOT = REPO_ROOT / 'StealthDiffusion_PixArtAlpha'
DIFFSR_ROOT = REPO_ROOT / 'DiffSRAttack'
if str(PIXART_ROOT) not in sys.path:
    sys.path.insert(0, str(PIXART_ROOT))
if str(DIFFSR_ROOT / 'evaluation') not in sys.path:
    sys.path.insert(0, str(DIFFSR_ROOT / 'evaluation'))

import diff_latent_attack
from ControlVAE import NewDecoder, NewEncoder
from eval_frequency_metrics import ResidualExtractor, l2_distance, mean_spectrum, rgb_to_gray, spectrum_log_amplitude
from eval_visual_quality import calculate_psnr, calculate_ssim
from main import load_diffusion_pipeline

CHECKPOINT_ROOT = REPO_ROOT / 'checkpoints'
DATASET_ROOT = REPO_ROOT / 'dataset'
PIXART_MODEL_PATH = CHECKPOINT_ROOT / 'hf_models' / 'pixart-alpha-xl-2-512x512'
SMOKE_FAKE_ROOT = DATASET_ROOT / 'smoke_fixed35' / 'fake'
REAL700_ROOT = DATASET_ROOT / 'original_genimage_eval_1400' / 'real'
RECON700_DIR = PIXART_ROOT / 'results' / 'pixart_ctrlvae_pixartproto_20260324' / 'recon_real_700' / 'flat_outputs'
EXP_ROOT = PIXART_ROOT / 'results' / 'pixart_ctrlvae_pixartproto_20260324' / 'diagnostics'
RECON_ONLY_ROOT = EXP_ROOT / 'recon_only_smoke35'
PROTO_ROOT = EXP_ROOT / 'prototype_compare'
OLD_CTRLVAE = CHECKPOINT_ROOT / 'Controlvae.pt'
NEW_CTRLVAE = CHECKPOINT_ROOT / 'Controlvae_pixartproto_20260324.pt'
NEW_PROTO = CHECKPOINT_ROOT / 'pixart_noise_prototype_20260324.pt'
OLDSTYLE_PROTO = PROTO_ROOT / 'oldstyle_cleanreal700_noise_prototype.pt'
RES = 224

PARAMS = {
    'act_fn': 'silu',
    'block_out_channels': [128, 256, 512, 512],
    'down_block_types': ['DownEncoderBlock2D'] * 4,
    'in_channels': 3,
    'latent_channels': 4,
    'layers_per_block': 2,
    'norm_num_groups': 32,
    'out_channels': 3,
    'sample_size': 512,
    'up_block_types': ['UpDecoderBlock2D'] * 4,
}
PARAMS_ENCODER = {
    'in_channels': PARAMS['in_channels'],
    'out_channels': PARAMS['latent_channels'],
    'down_block_types': PARAMS['down_block_types'],
    'block_out_channels': PARAMS['block_out_channels'],
    'layers_per_block': PARAMS['layers_per_block'],
    'act_fn': PARAMS['act_fn'],
    'norm_num_groups': PARAMS['norm_num_groups'],
    'double_z': True,
}
PARAMS_DECODER = {
    'in_channels': PARAMS['latent_channels'],
    'out_channels': PARAMS['out_channels'],
    'up_block_types': PARAMS['up_block_types'],
    'block_out_channels': PARAMS['block_out_channels'],
    'layers_per_block': PARAMS['layers_per_block'],
    'act_fn': PARAMS['act_fn'],
    'norm_num_groups': PARAMS['norm_num_groups'],
}
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.JPEG', '.JPG', '.PNG'}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def iter_images(root: Path) -> list[Path]:
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in IMG_EXTS]
    files.sort(key=lambda p: p.as_posix())
    return files


def infer_subset(path: Path, fake_root: Path) -> str:
    rel = path.relative_to(fake_root)
    return rel.parts[0].lower()


def load_resized_image(path: Path, res: int = RES) -> Image.Image:
    return Image.open(path).convert('RGB').resize((res, res), resample=Image.LANCZOS)


def tensor_to_u8(t: torch.Tensor) -> np.ndarray:
    t = t.detach().float().cpu().clamp(-1.0, 1.0)
    if t.ndim == 4:
        t = t[0]
    arr = (t.permute(1, 2, 0) / 2 + 0.5).clamp(0, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def u8_to_01(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def fourier_l2_pair(clean_u8: np.ndarray, recon_u8: np.ndarray) -> float:
    clean_spec = spectrum_log_amplitude(rgb_to_gray(u8_to_01(clean_u8)))
    recon_spec = spectrum_log_amplitude(rgb_to_gray(u8_to_01(recon_u8)))
    return l2_distance(recon_spec, clean_spec)


def load_control_branch(pipe, ckpt_path: Path, device: torch.device) -> tuple[NewEncoder, NewDecoder]:
    state = torch.load(ckpt_path, map_location=device)
    encoder = NewEncoder(**PARAMS_ENCODER)
    encoder.load_state_dict(state['encoder'], strict=False)
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    encoder.requires_grad_(False)

    decoder = NewDecoder(**PARAMS_DECODER)
    decoder.load_state_dict(pipe.vae.decoder.state_dict(), strict=False)
    decoder = decoder.to(device=device, dtype=torch.float32).eval()
    decoder.requires_grad_(False)
    return encoder, decoder


@torch.no_grad()
def reconstruct_base(pipe, image: Image.Image, res: int = RES) -> np.ndarray:
    vae_param = next(pipe.vae.parameters())
    x = diff_latent_attack.preprocess(image, res, device=vae_param.device, dtype=vae_param.dtype)
    posterior = pipe.vae.encode(x).latent_dist
    latents = diff_latent_attack.get_scaling_factor(pipe) * posterior.mode()
    recon = pipe.vae.decode(latents / diff_latent_attack.get_scaling_factor(pipe))['sample']
    return tensor_to_u8(recon)


@torch.no_grad()
def reconstruct_controlvae(pipe, image: Image.Image, branch: tuple[NewEncoder, NewDecoder], res: int = RES) -> np.ndarray:
    encoder, decoder = branch
    vae_param = next(pipe.vae.parameters())
    device = vae_param.device
    x_vae = diff_latent_attack.preprocess(image, res, device=device, dtype=vae_param.dtype)
    posterior = pipe.vae.encode(x_vae).latent_dist
    latents = diff_latent_attack.get_scaling_factor(pipe) * posterior.mode()
    x_ctrl = x_vae.to(dtype=torch.float32)
    down_features = encoder(x_ctrl)[1]
    decode_latents = pipe.vae.post_quant_conv((latents / diff_latent_attack.get_scaling_factor(pipe)).to(dtype=vae_param.dtype))
    recon = decoder(decode_latents.to(dtype=torch.float32), down_features)
    return tensor_to_u8(recon)


def save_recon_u8(arr: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = ['psnr', 'ssim', 'fourier_l2_pair']
    out = {}
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f'{key}_mean'] = float(vals.mean())
        out[f'{key}_std'] = float(vals.std())
    return out


def radial_profile(arr: np.ndarray) -> list[float]:
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = int(rr.max())
    out = []
    for radius in range(max_r + 1):
        mask = rr == radius
        out.append(float(arr[mask].mean()) if mask.any() else float('nan'))
    return out


def signed_to_rgb(arr_chw: np.ndarray) -> np.ndarray:
    vmax = float(np.max(np.abs(arr_chw)))
    vmax = vmax if vmax > 0 else 1.0
    norm = np.clip(arr_chw / vmax, -1.0, 1.0)
    rgb = ((norm.transpose(1, 2, 0) * 0.5) + 0.5) * 255.0
    return rgb.round().astype(np.uint8)


def positive_to_gray(arr_hw: np.ndarray) -> np.ndarray:
    amin = float(arr_hw.min())
    amax = float(arr_hw.max())
    denom = (amax - amin) if amax > amin else 1.0
    gray = ((arr_hw - amin) / denom) * 255.0
    g = gray.round().astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def save_grid(path: Path, labeled_images: list[tuple[str, np.ndarray]]) -> None:
    tiles = []
    for label, arr in labeled_images:
        img = Image.fromarray(arr)
        canvas = Image.new('RGB', (img.width, img.height + 24), color=(255, 255, 255))
        canvas.paste(img, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), label, fill=(0, 0, 0))
        tiles.append(canvas)
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    grid = Image.new('RGB', (width, height), color=(255, 255, 255))
    x = 0
    for tile in tiles:
        grid.paste(tile, (x, 0))
        x += tile.width
    ensure_dir(path.parent)
    grid.save(path)


def run_recon_only() -> None:
    out_root = RECON_ONLY_ROOT
    ensure_dir(out_root)
    smoke_files = iter_images(SMOKE_FAKE_ROOT)
    if len(smoke_files) != 35:
        raise ValueError(f'Expected 35 smoke files, got {len(smoke_files)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = load_diffusion_pipeline(str(PIXART_MODEL_PATH), str(device))
    pipe.set_progress_bar_config(disable=True)
    old_branch = load_control_branch(pipe, OLD_CTRLVAE, device)
    new_branch = load_control_branch(pipe, NEW_CTRLVAE, device)

    rows_by_mode: dict[str, list[dict[str, object]]] = {'pixart_vae': [], 'ctrlvae_old': [], 'ctrlvae_new': []}
    mode_roots = {
        'pixart_vae': out_root / 'outputs' / 'pixart_vae_surS',
        'ctrlvae_old': out_root / 'outputs' / 'ctrlvae_old_surS',
        'ctrlvae_new': out_root / 'outputs' / 'ctrlvae_new_surS',
    }
    for p in mode_roots.values():
        ensure_dir(p / 'real')
        ensure_dir(p / 'fake')

    clean_mean, _ = mean_spectrum(smoke_files, res=RES, residual_extractor=None)
    real_mean, _ = mean_spectrum(iter_images(REAL700_ROOT), res=RES, residual_extractor=None)

    for img_path in tqdm(smoke_files, desc='recon_only_smoke35'):
        subset = infer_subset(img_path, SMOKE_FAKE_ROOT)
        rel = img_path.relative_to(SMOKE_FAKE_ROOT)
        image = load_resized_image(img_path)
        clean_u8 = np.array(image, dtype=np.uint8)
        outputs = {
            'pixart_vae': reconstruct_base(pipe, image, res=RES),
            'ctrlvae_old': reconstruct_controlvae(pipe, image, old_branch, res=RES),
            'ctrlvae_new': reconstruct_controlvae(pipe, image, new_branch, res=RES),
        }
        for mode, recon_u8 in outputs.items():
            out_path = mode_roots[mode] / 'fake' / rel
            save_recon_u8(recon_u8, out_path)
            rows_by_mode[mode].append({
                'subset': subset,
                'clean_path': str(img_path),
                'recon_path': str(out_path),
                'psnr': calculate_psnr(clean_u8, recon_u8),
                'ssim': calculate_ssim(clean_u8, recon_u8),
                'fourier_l2_pair': fourier_l2_pair(clean_u8, recon_u8),
            })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {}
    for mode, rows in rows_by_mode.items():
        write_csv(out_root / f'{mode}_per_image.csv', ['subset', 'clean_path', 'recon_path', 'psnr', 'ssim', 'fourier_l2_pair'], rows)
        recon_files = [Path(r['recon_path']) for r in rows]
        recon_mean, _ = mean_spectrum(recon_files, res=RES, residual_extractor=None)
        subset_rows = []
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[str(row['subset'])].append(row)
        for subset, subset_rows_src in sorted(groups.items()):
            s = summarize_rows(subset_rows_src)
            subset_rows.append({'subset': subset, 'n': len(subset_rows_src), **s})
        write_csv(out_root / f'{mode}_by_subset.csv', ['subset', 'n', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std', 'fourier_l2_pair_mean', 'fourier_l2_pair_std'], subset_rows)
        summary[mode] = {
            'n': len(rows),
            **summarize_rows(rows),
            'dataset_mean_logamp_l2_to_clean': l2_distance(recon_mean, clean_mean),
            'dataset_mean_logamp_l2_to_real700': l2_distance(recon_mean, real_mean),
        }
    save_json(out_root / 'summary.json', summary)
    save_json(out_root / 'run_meta.json', {
        'problem': 'Diagnose whether the new PixArt-specific ControlVAE is already spectrally poor under pure encode/decode, before LAO enters.',
        'input_root': str(SMOKE_FAKE_ROOT),
        'modes': ['pixart_vae', 'ctrlvae_old', 'ctrlvae_new'],
        'backbone': 'pixart',
        'res': RES,
        'controlvae_old_ckpt': str(OLD_CTRLVAE),
        'controlvae_new_ckpt': str(NEW_CTRLVAE),
    })


def compute_dataset_prototype(files: list[Path], extractor: ResidualExtractor) -> tuple[torch.Tensor, dict[str, float]]:
    acc = None
    norms = []
    abs_means = []
    stds = []
    for path in tqdm(files, desc='residual_stats'):
        rgb = np.array(load_resized_image(path), dtype=np.uint8).astype(np.float32) / 255.0
        residual = extractor(rgb)
        chw = np.transpose(residual, (2, 0, 1))
        acc = chw if acc is None else acc + chw
        norms.append(float(np.linalg.norm(chw.reshape(-1), ord=2)))
        abs_means.append(float(np.abs(chw).mean()))
        stds.append(float(chw.std()))
    proto = torch.from_numpy((acc / len(files)).astype(np.float32)).unsqueeze(0)
    stats = {
        'n': len(files),
        'residual_norm_mean': float(np.mean(norms)),
        'residual_norm_std': float(np.std(norms)),
        'residual_abs_mean_mean': float(np.mean(abs_means)),
        'residual_abs_mean_std': float(np.std(abs_means)),
        'residual_std_mean': float(np.mean(stds)),
        'residual_std_std': float(np.std(stds)),
    }
    return proto, stats


def prototype_stats(proto_tensor: torch.Tensor) -> dict[str, object]:
    chw = proto_tensor.detach().float().cpu().squeeze(0).numpy()
    gray = rgb_to_gray(np.transpose(chw, (1, 2, 0)))
    fft = np.fft.fftshift(np.fft.fft2(gray))
    amp = np.abs(fft)
    logamp = np.log1p(amp).astype(np.float32)
    h, w = logamp.shape
    cy, cx = h // 2, w // 2
    center = logamp[max(0, cy - 5): cy + 6, max(0, cx - 5): cx + 6]
    return {
        'shape': list(proto_tensor.shape),
        'l2_norm': float(np.linalg.norm(chw.reshape(-1), ord=2)),
        'mean_abs': float(np.abs(chw).mean()),
        'std': float(chw.std()),
        'max_abs': float(np.abs(chw).max()),
        'fft_energy_mean': float(np.mean(amp ** 2)),
        'fft_logamp_mean': float(logamp.mean()),
        'fft_logamp_std': float(logamp.std()),
        'fft_logamp_max': float(logamp.max()),
        'center_peak_mean': float(center.mean()),
        'center_peak_max': float(center.max()),
        'center_peak_overall_ratio': float(center.mean() / max(logamp.mean(), 1e-8)),
        'radial_profile': radial_profile(logamp),
        'spatial_rgb_u8': signed_to_rgb(chw),
        'fft_rgb_u8': positive_to_gray(logamp),
    }


def run_prototype_compare() -> None:
    out_root = PROTO_ROOT
    ensure_dir(out_root)
    extractor = ResidualExtractor(CHECKPOINT_ROOT / 'dncnn_color_blind.pth')
    real_files = iter_images(REAL700_ROOT)
    recon_files = sorted(RECON700_DIR.glob('*_adv_image.png'))
    if len(real_files) != 700:
        raise ValueError(f'Expected 700 real files, got {len(real_files)}')
    if len(recon_files) != 700:
        raise ValueError(f'Expected 700 recon files, got {len(recon_files)}')

    if OLDSTYLE_PROTO.exists():
        old_proto = torch.load(OLDSTYLE_PROTO, map_location='cpu').float()
        _, clean_real_stats = compute_dataset_prototype([], extractor) if False else (None, None)
    else:
        old_proto, clean_real_stats = compute_dataset_prototype(real_files, extractor)
        torch.save(old_proto, OLDSTYLE_PROTO)
    if clean_real_stats is None:
        _, clean_real_stats = compute_dataset_prototype(real_files, extractor)

    _, recon_stats = compute_dataset_prototype(recon_files, extractor)
    new_proto = torch.load(NEW_PROTO, map_location='cpu').float()
    old_stats = prototype_stats(old_proto)
    new_stats = prototype_stats(new_proto)

    save_grid(out_root / 'prototype_spatial_grid.png', [
        ('oldstyle_clean_real700_spatial', old_stats.pop('spatial_rgb_u8')),
        ('new_pixart_proto_spatial', new_stats.pop('spatial_rgb_u8')),
    ])
    save_grid(out_root / 'prototype_fft_grid.png', [
        ('oldstyle_clean_real700_fft', old_stats.pop('fft_rgb_u8')),
        ('new_pixart_proto_fft', new_stats.pop('fft_rgb_u8')),
    ])

    radial_len = max(len(old_stats['radial_profile']), len(new_stats['radial_profile']))
    rows = []
    for i in range(radial_len):
        rows.append({
            'radius': i,
            'oldstyle_clean_real700': old_stats['radial_profile'][i] if i < len(old_stats['radial_profile']) else '',
            'new_pixart_proto': new_stats['radial_profile'][i] if i < len(new_stats['radial_profile']) else '',
        })
    write_csv(out_root / 'prototype_radial_profiles.csv', ['radius', 'oldstyle_clean_real700', 'new_pixart_proto'], rows)

    report = {
        'problem': 'Check whether the new PixArt-specific prototype is actually better matched, or simply larger / sharper / more artifact-heavy than the old-style baseline prototype.',
        'oldstyle_baseline_note': 'The original prototype artifact was not stored, so this baseline is re-estimated with the original train_vae.py rule on clean real700 images.',
        'oldstyle_clean_real700': old_stats,
        'new_pixart_proto': new_stats,
        'clean_real700_residual_distribution': clean_real_stats,
        'pixart_recon700_residual_distribution': recon_stats,
        'new_over_old': {
            'l2_norm_ratio': float(new_stats['l2_norm'] / max(old_stats['l2_norm'], 1e-8)),
            'mean_abs_ratio': float(new_stats['mean_abs'] / max(old_stats['mean_abs'], 1e-8)),
            'std_ratio': float(new_stats['std'] / max(old_stats['std'], 1e-8)),
            'fft_energy_ratio': float(new_stats['fft_energy_mean'] / max(old_stats['fft_energy_mean'], 1e-8)),
            'center_peak_ratio_ratio': float(new_stats['center_peak_overall_ratio'] / max(old_stats['center_peak_overall_ratio'], 1e-8)),
        },
    }
    save_json(out_root / 'prototype_compare_report.json', report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['recon_only', 'prototype_compare', 'all'])
    args = ap.parse_args()
    if args.action in {'recon_only', 'all'}:
        run_recon_only()
    if args.action in {'prototype_compare', 'all'}:
        run_prototype_compare()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
