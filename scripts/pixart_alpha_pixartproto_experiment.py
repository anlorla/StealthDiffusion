#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path('/root/gpufree-data')
PIXART_ROOT = REPO_ROOT / 'StealthDiffusion_PixArtAlpha'
if str(PIXART_ROOT) not in sys.path:
    sys.path.insert(0, str(PIXART_ROOT))

import diff_latent_attack
from main import load_diffusion_pipeline
DIFFSR_ROOT = REPO_ROOT / 'DiffSRAttack'
DATASET_ROOT = REPO_ROOT / 'dataset'
CHECKPOINT_ROOT = REPO_ROOT / 'checkpoints'
PIXART_PYTHON = REPO_ROOT / 'conda_envs' / 'pixart_alpha' / 'bin' / 'python'
PIXART_MODEL_PATH = CHECKPOINT_ROOT / 'hf_models' / 'pixart-alpha-xl-2-512x512'
SD21_LOCAL_PATH = Path('/root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06')
CLEAN_DATAROOT = DATASET_ROOT / 'original_genimage_eval_1400'
REAL_ROOT = CLEAN_DATAROOT / 'real'
SMOKE35_FAKE_ROOT = DATASET_ROOT / 'smoke_fixed35' / 'fake'
ATTACK_DATASETS_ROOT = DATASET_ROOT / 'pixart_alpha' / 'attack_datasets'
EXPERIMENT_DATE = '20260324'
EXPERIMENT_ROOT = PIXART_ROOT / 'results' / f'pixart_ctrlvae_pixartproto_{EXPERIMENT_DATE}'
RECON_ROOT = EXPERIMENT_ROOT / 'recon_real_700'
TRAIN_ROOT = EXPERIMENT_ROOT / 'train_controlvae'
SMOKE_ROOT = EXPERIMENT_ROOT / 'smoke35_eval_surS'
SUITE_SPEC_JSON = EXPERIMENT_ROOT / 'suite_spec.json'
SUITE_SPEC_MD = EXPERIMENT_ROOT / 'suite_spec.md'
RECON_SAVE_DIR = RECON_ROOT / 'flat_outputs'
RECON_INTERMEDIATE_ROOT = RECON_ROOT / 'intermediates'
RECON_MANIFEST = RECON_ROOT / 'recon_manifest.txt'
RECON_META_JSON = RECON_ROOT / 'run_meta.json'
RECON_META_MD = RECON_ROOT / 'run_meta.md'
RECON_STDOUT = RECON_ROOT / 'run_stdout.log'
RECON_PAIRS_CSV = RECON_ROOT / 'recon_pairs.csv'
RECON_SUMMARY_JSON = RECON_ROOT / 'summary_metrics.json'
TRAIN_ARTIFACTS = TRAIN_ROOT / 'train_artifacts'
TRAIN_META_JSON = TRAIN_ROOT / 'run_meta.json'
TRAIN_META_MD = TRAIN_ROOT / 'run_meta.md'
TRAIN_STDOUT = TRAIN_ROOT / 'train_stdout.log'
TRAIN_INTERNAL_LOG = TRAIN_ROOT / 'train_vae_internal.log'
TRAIN_CHECKPOINT = CHECKPOINT_ROOT / f'Controlvae_pixartproto_{EXPERIMENT_DATE}.pt'
TRAIN_NOISE_PROTOTYPE = CHECKPOINT_ROOT / f'pixart_noise_prototype_{EXPERIMENT_DATE}.pt'
SMOKE_RUN_DIR = SMOKE_ROOT / 'runs' / 'pixart_ctrlvae_pixartproto_surS'
SMOKE_META_JSON = SMOKE_ROOT / 'run_meta.json'
SMOKE_META_MD = SMOKE_ROOT / 'run_meta.md'
SMOKE_STDOUT = SMOKE_ROOT / 'run_stdout.log'
SMOKE_OUTPUT_DATAROOT = ATTACK_DATASETS_ROOT / 'smoke35' / f'PixArtAlpha_smoke35_adv_ctrlvae_pixartproto_surS_{EXPERIMENT_DATE}'
SMOKE_RECORD_JSON = SMOKE_ROOT / 'record_candidate.json'
SMOKE_RECORD_MD = SMOKE_ROOT / 'record_candidate.md'
EXPERIMENT_RECORD_CSV = DIFFSR_ROOT / 'results' / 'experiment_record_log.csv'

FIELDNAMES = [
    'record_order', 'record_id', 'status', 'project', 'family', 'experiment_name', 'date_hint',
    'scope', 'data_n', 'result_path', 'backbone', 'surrogate', 'controlvae', 'iterations',
    't_start', 'lr', 'lambda_perc', 'primary_change', 'whitebox_asr', 'blackbox_mean_asr',
    'asr_to_E', 'asr_to_R', 'asr_to_D', 'asr_to_S', 'lpips', 'psnr', 'ssim',
    'fourier_l2_to_real', 'residual_fourier_l2_to_real', 'attack_mean_psnr', 'attack_mean_ssim',
    'compared_to', 'new_information', 'evidence'
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding='utf-8')


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fmt(x) -> str:
    if x == '' or x is None:
        return ''
    if isinstance(x, float):
        return f'{x:.6f}'
    return str(x)


def run_cmd(cmd: list[str], cwd: Path, stdout_path: Path | None = None) -> None:
    print('+', ' '.join(str(x) for x in cmd), flush=True)
    if stdout_path is None:
        subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)
        return
    ensure_dir(stdout_path.parent)
    with stdout_path.open('a', encoding='utf-8') as f:
        f.write('+ ' + ' '.join(str(x) for x in cmd) + '\n')
        f.flush()
        subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True, stdout=f, stderr=subprocess.STDOUT)


def copy_text_like(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    ensure_dir(link_path.parent)
    rel = os.path.relpath(target, start=link_path.parent)
    os.symlink(rel, link_path)


def suite_spec() -> dict[str, object]:
    return {
        'experiment_id': f'pixart_ctrlvae_pixartproto_{EXPERIMENT_DATE}',
        'problem': 'Validate whether PixArt+ControlVAE regression is mainly caused by a noise prototype mismatch.',
        'hypothesis': 'If the noise prototype is re-estimated from PixArt reconstruction outputs, Fourier distance should improve without sacrificing blackbox transfer as much as the current ControlVAE baseline.',
        'backbone': 'pixart',
        'pipeline': [
            'collect PixArt reconstruction-only outputs on real-700',
            'estimate PixArt-specific noise prototype with original DnCNN-based code path',
            'train ControlVAE with original train_vae.py settings',
            'run smoke_fixed35 surrogate-S evaluation with retrained weights',
            'export record candidate for experiment_record_log.csv',
        ],
        'candidate_modules': {
            'backbone': ['PixArtAlphaPipeline.transformer (DiT)'],
            'vae_related': ['vae encoder', 'vae decoder', 'ControlVAE NewEncoder'],
            'detectors': ['efficientnet-b0', 'resnet50', 'deit', 'swin-t'],
            'prototype_loss': ['DnCNN residual extractor', 'prototype-based NPL'],
        },
        'metrics': ['whitebox_asr', 'blackbox_mean_asr', 'fourier_l2_to_real', 'psnr', 'lpips', 'ssim'],
        'baselines': ['pixart_noctrl_700_surS', 'pixart_ctrl_700_surS'],
        'inputs': {
            'real_700': str(REAL_ROOT),
            'smoke35_fake': str(SMOKE35_FAKE_ROOT),
            'clean_dataroot': str(CLEAN_DATAROOT),
        },
        'outputs': {
            'experiment_root': str(EXPERIMENT_ROOT),
            'reconstruction_root': str(RECON_ROOT),
            'training_root': str(TRAIN_ROOT),
            'smoke_eval_root': str(SMOKE_ROOT),
            'prototype': str(TRAIN_NOISE_PROTOTYPE),
            'checkpoint': str(TRAIN_CHECKPOINT),
            'smoke_attack_dataroot': str(SMOKE_OUTPUT_DATAROOT),
        },
        'logging_policy': {
            'experiment_record_csv': str(EXPERIMENT_RECORD_CSV),
            'record_backbone': 'pixart',
            'record_surrogate': 'S',
            'record_controlvae': 1,
            'record_primary_change': 'pixart_specific_noise_prototype',
        },
        'training_choice_this_run': {
            'principle': 'Use the original author code path first; do not alter the training target modules yet.',
            'actual_behavior': 'train_vae.py trains the full NewEncoder branch under the original script, while decoder/vae/text encoder/backbone stay frozen.',
            'ablation_saved_for_later': ['train zero_conv only', 'resume vs no-resume', 'old prototype vs PixArt prototype'],
        },
    }


def suite_spec_md(spec: dict[str, object]) -> str:
    return '\n'.join([
        f"# {spec['experiment_id']}",
        '',
        '## Experiment Definition',
        f"- Problem: {spec['problem']}",
        f"- Hypothesis: {spec['hypothesis']}",
        f"- Backbone: {spec['backbone']}",
        '- Pipeline:',
        *[f"  - {x}" for x in spec['pipeline']],
        '- Candidate modules:',
        *[f"  - {k}: {', '.join(v)}" for k, v in spec['candidate_modules'].items()],
        '',
        '## Metrics',
        f"- Metrics: {', '.join(spec['metrics'])}",
        f"- Baselines: {', '.join(spec['baselines'])}",
        '',
        '## Inputs/Outputs',
        f"- Real input: {spec['inputs']['real_700']}",
        f"- Smoke input: {spec['inputs']['smoke35_fake']}",
        f"- Experiment root: {spec['outputs']['experiment_root']}",
        f"- Prototype: {spec['outputs']['prototype']}",
        f"- Checkpoint: {spec['outputs']['checkpoint']}",
        '',
        '## Logging Policy',
        f"- Experiment CSV: {spec['logging_policy']['experiment_record_csv']}",
        f"- Record backbone: {spec['logging_policy']['record_backbone']}",
        f"- Primary change: {spec['logging_policy']['record_primary_change']}",
        '',
        '## Current Training Choice',
        f"- Principle: {spec['training_choice_this_run']['principle']}",
        f"- Actual behavior: {spec['training_choice_this_run']['actual_behavior']}",
        f"- Later ablations: {', '.join(spec['training_choice_this_run']['ablation_saved_for_later'])}",
        '',
    ])


def write_suite_spec() -> None:
    spec = suite_spec()
    write_json(SUITE_SPEC_JSON, spec)
    write_text(SUITE_SPEC_MD, suite_spec_md(spec))


def reconstruction_meta() -> dict[str, object]:
    return {
        'stage': 'reconstruct_real_700',
        'problem': 'Collect PixArt reconstruction-only outputs on the real-700 set as prototype-estimation inputs.',
        'pipeline': [
            'original PixArt DDIM inversion / denoise / VAE decode helpers',
            'real-image input only',
            'DDIM inversion',
            'DiT denoise',
            'standard VAE decode',
            'no adversarial perturbation (guidance=0, iterations=0)',
        ],
        'candidate_modules': {
            'enabled': ['PixArtAlphaPipeline.transformer (DiT)', 'vae'],
            'disabled': ['ControlVAE NewEncoder', 'ControlVAE NewDecoder', 'latent optimization', 'non-zero PGD perturbation'],
        },
        'metrics': ['attack_mean_psnr', 'attack_mean_ssim'],
        'input_dataset': str(REAL_ROOT),
        'clean_dataroot': str(CLEAN_DATAROOT),
        'output_dataset': str(RECON_SAVE_DIR),
        'output_logs': str(RECON_ROOT),
        'output_weights': 'none',
        'backbone': 'pixart',
        'surrogate': 'S',
        'controlvae': 0,
        'primary_change': 'pixart_reconstruction_only_real700',
    }


def train_meta() -> dict[str, object]:
    return {
        'stage': 'train_controlvae_pixartproto',
        'problem': 'Train a ControlVAE checkpoint with a PixArt-specific noise prototype while keeping the original train_vae.py behavior unchanged.',
        'pipeline': [
            'original train_vae.py',
            'reconstruction manifest as training data',
            'DnCNN residual averaging for prototype estimation',
            'prototype-based NPL enabled',
            'original NewEncoder training path',
        ],
        'candidate_modules': {
            'enabled': ['ControlVAE NewEncoder', 'DnCNN prototype estimator', 'LPIPS', 'L1', 'NPL'],
            'frozen_by_original_code': ['decoder', 'vae', 'text_encoder', 'backbone/denoiser'],
        },
        'metrics': ['train losses in internal log'],
        'input_dataset': str(RECON_MANIFEST),
        'output_dataset': str(TRAIN_ARTIFACTS),
        'output_logs': str(TRAIN_ROOT),
        'output_weights': str(TRAIN_CHECKPOINT),
        'output_noise_prototype': str(TRAIN_NOISE_PROTOTYPE),
        'backbone': 'pixart',
        'surrogate': 'S',
        'controlvae': 1,
        'primary_change': 'pixart_specific_noise_prototype',
        'training_choice': 'Use original train_vae.py behavior; do not yet isolate zero-conv-only updates.',
    }


def smoke_meta() -> dict[str, object]:
    return {
        'stage': 'smoke35_eval_surS',
        'problem': 'Evaluate the retrained PixArt+ControlVAE checkpoint on the fixed smoke35 subset under surrogate S.',
        'pipeline': [
            'original PixArt main.py attack path',
            'DiT backbone + DDIM inversion + LAO',
            'ControlVAE NewEncoder-conditioned decode',
            'DiffSRAttack evaluation scripts for transfer / visual / frequency metrics',
        ],
        'candidate_modules': {
            'enabled': ['PixArtAlphaPipeline.transformer (DiT)', 'vae', 'ControlVAE NewEncoder', 'detectors E/R/D/S'],
            'disabled': ['training-time ablations'],
        },
        'metrics': ['whitebox_asr', 'blackbox_mean_asr', 'fourier_l2_to_real', 'psnr', 'lpips', 'ssim'],
        'input_dataset': str(SMOKE35_FAKE_ROOT),
        'clean_dataroot': str(CLEAN_DATAROOT),
        'output_dataset': str(SMOKE_OUTPUT_DATAROOT),
        'output_logs': str(SMOKE_ROOT),
        'output_weights': str(TRAIN_CHECKPOINT),
        'backbone': 'pixart',
        'surrogate': 'S',
        'controlvae': 1,
        'primary_change': 'pixart_specific_noise_prototype',
        'compared_to': 'pixart_noctrl_700_surS|pixart_ctrl_700_surS',
    }


def write_stage_meta(meta: dict[str, object], json_path: Path, md_path: Path) -> None:
    write_json(json_path, meta)
    lines = [f"# {meta['stage']}", '', '## Experiment Definition']
    lines.append(f"- Problem: {meta['problem']}")
    lines.append('- Pipeline:')
    lines.extend([f"  - {x}" for x in meta['pipeline']])
    lines.append('- Candidate modules:')
    for k, v in meta['candidate_modules'].items():
        lines.append(f"  - {k}: {', '.join(v)}")
    lines.extend([
        '', '## Metrics', f"- Metrics: {', '.join(meta['metrics'])}",
        '', '## Inputs/Outputs',
        f"- Input dataset: {meta['input_dataset']}",
        f"- Output dataset: {meta['output_dataset']}",
        f"- Output logs: {meta['output_logs']}",
        f"- Output weights: {meta.get('output_weights', 'none')}",
    ])
    if meta.get('clean_dataroot'):
        lines.append(f"- Clean dataroot: {meta['clean_dataroot']}")
    if meta.get('output_noise_prototype'):
        lines.append(f"- Output noise prototype: {meta['output_noise_prototype']}")
    lines.extend([
        '', '## Logging',
        f"- Backbone field: {meta['backbone']}",
        f"- Surrogate field: {meta['surrogate']}",
        f"- ControlVAE field: {meta['controlvae']}",
        f"- Primary change: {meta['primary_change']}",
        '',
    ])
    write_text(md_path, '\n'.join(lines))


def build_reconstruction_cmd() -> list[str]:
    return [
        str(PIXART_PYTHON), str(PIXART_ROOT / 'main.py'),
        '--output_mode', 'flat',
        '--save_dir', str(RECON_SAVE_DIR),
        '--clean_dataroot', str(CLEAN_DATAROOT),
        '--images_root', str(REAL_ROOT),
        '--pretrained_diffusion_path', str(PIXART_MODEL_PATH),
        '--model_name', 'S,E,R,D',
        '--dataset_name', 'ours_try',
        '--is_encoder', '0',
        '--diffusion_steps', '20',
        '--start_step', '18',
        '--iterations', '0',
        '--lr', '1e-3',
        '--res', '224',
        '--guidance', '0.0',
        '--eps', '0.0',
        '--attack_loss_weight', '0',
        '--pgd_steps', '1',
        '--lambda_l1', '0.0',
        '--lambda_lpips', '0.0',
        '--lambda_latent', '0.0',
        '--attack_only_fake', '0',
        '--seed', '42',
        '--save_intermediates',
        '--intermediate_root', str(RECON_INTERMEDIATE_ROOT),
    ]




def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_real_images(root: Path) -> list[Path]:
    images = sorted(p for p in root.rglob('*') if p.is_file() and is_image_file(p))
    if len(images) != 700:
        raise ValueError(f'Expected 700 real images under {root}, got {len(images)}')
    if any(p.is_symlink() for p in images):
        raise ValueError(f'Real input set contains symlinks under {root}')
    return images


def load_resized_rgb(path: Path, res: int) -> Image.Image:
    image = Image.open(path).convert('RGB')
    return image.resize((res, res), resample=Image.LANCZOS)


def tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image_tensor = image_tensor.detach().float().cpu().clamp(-1.0, 1.0)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image = (image_tensor.permute(1, 2, 0) / 2 + 0.5).clamp(0, 1).numpy()
    return (image * 255.0).round().astype(np.uint8)


def save_tensor_image(image_tensor: torch.Tensor, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(tensor_to_uint8(image_tensor)).save(path)


@torch.no_grad()
def run_pixart_ddim_recon(pipe, image: Image.Image, res: int, num_inference_steps: int, context: dict[str, torch.Tensor]) -> tuple[torch.Tensor, float, float]:
    exec_device = diff_latent_attack.get_pipeline_execution_device(pipe)
    pipe.scheduler.set_timesteps(num_inference_steps, device=exec_device)
    pipe.inverse_scheduler.set_timesteps(num_inference_steps, device=exec_device)

    latent = diff_latent_attack.encoder(image, pipe, res=res, sample_posterior=False)

    for t in pipe.inverse_scheduler.timesteps[1:]:
        model_output = diff_latent_attack.get_model_prediction(pipe, latent, context, t, guidance_scale=0.0)
        latent = pipe.inverse_scheduler.step(model_output, t, latent)['prev_sample']

    for t in pipe.scheduler.timesteps:
        latent = diff_latent_attack.diffusion_step(pipe, latent, context, t, guidance_scale=0.0)

    decode_latents = (latent / diff_latent_attack.get_scaling_factor(pipe)).to(dtype=next(pipe.vae.parameters()).dtype)
    recon = pipe.vae.decode(decode_latents)['sample']

    image_uint8 = np.array(image, dtype=np.uint8)
    recon_uint8 = tensor_to_uint8(recon)
    psnr = float(diff_latent_attack.calculate_psnr(image_uint8, recon_uint8))
    ssim = float(diff_latent_attack.calculate_ssim(image_uint8, recon_uint8))
    return recon, psnr, ssim


def write_reconstruction_log(lines: list[str]) -> None:
    ensure_dir(RECON_STDOUT.parent)
    with RECON_STDOUT.open('a', encoding='utf-8') as f:
        for line in lines:
            f.write(line.rstrip() + '\n')


def run_reconstruct_batch() -> None:
    ensure_dir(RECON_SAVE_DIR)
    images = list_real_images(REAL_ROOT)
    if any(RECON_SAVE_DIR.glob('*_adv_image.png')):
        raise FileExistsError(f'Reconstruction outputs already exist under {RECON_SAVE_DIR}')

    write_reconstruction_log([
        '# reconstruct_real_700 started',
        f'python={PIXART_PYTHON}',
        f'pixart_path={PIXART_MODEL_PATH}',
        f'real_root={REAL_ROOT}',
        f'output_dir={RECON_SAVE_DIR}',
        'reconstruction_impl=direct_pixart_ddim_helper',
        'note=main.py real-only no-attack path previously failed; this stage now reuses the same PixArt DDIM/VAE helper path directly for stable batch reconstruction.',
    ])

    pipe = load_diffusion_pipeline(str(PIXART_MODEL_PATH), 'cuda' if torch.cuda.is_available() else 'cpu')
    pipe.set_progress_bar_config(disable=True)
    res = 224
    num_steps = 20
    shared_context = diff_latent_attack.build_context(pipe, [''], res=res)
    moved_modules = diff_latent_attack.offload_unused_conditioning_modules(pipe)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    rows: list[dict[str, object]] = []
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []

    for idx, src_path in enumerate(tqdm(images, desc='pixart_reconstruct_real700')):
        image = load_resized_rgb(src_path, res=res)
        recon, psnr, ssim = run_pixart_ddim_recon(pipe, image, res=res, num_inference_steps=num_steps, context=shared_context)
        out_path = RECON_SAVE_DIR / f'{idx:04d}_{src_path.stem}_adv_image.png'
        save_tensor_image(recon, out_path)
        rows.append({
            'index': idx,
            'source_path': str(src_path),
            'recon_path': str(out_path),
            'psnr': f'{psnr:.6f}',
            'ssim': f'{ssim:.6f}',
        })
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)

        del image, recon
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    with RECON_PAIRS_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'source_path', 'recon_path', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'count': len(rows),
        'mean_psnr': float(sum(psnr_vals) / len(psnr_vals)),
        'mean_ssim': float(sum(ssim_vals) / len(ssim_vals)),
        'shared_context_prompt': '',
        'res': res,
        'diffusion_steps': num_steps,
        'moved_modules_after_context_build': moved_modules,
    }
    write_json(RECON_SUMMARY_JSON, summary)
    write_reconstruction_log([
        f'count={summary["count"]}',
        f'mean_psnr={summary["mean_psnr"]:.6f}',
        f'mean_ssim={summary["mean_ssim"]:.6f}',
        f'recon_pairs_csv={RECON_PAIRS_CSV}',
    ])


def build_train_cmd() -> list[str]:
    pretrained_path = SD21_LOCAL_PATH if SD21_LOCAL_PATH.exists() else Path('stabilityai/stable-diffusion-2-1-base')
    return [
        str(PIXART_PYTHON), str(PIXART_ROOT / 'train_vae.py'),
        '--name', f'controlvae_pixartproto_real700_{EXPERIMENT_DATE}',
        '--pipeline_type', 'sd',
        '--pretrained_diffusion_path', str(pretrained_path),
        '--manifest', str(RECON_MANIFEST),
        '--iter', '20',
        '--batch_size', '8',
        '--size', '224',
        '--seed', EXPERIMENT_DATE,
        '--enable_npl', '1',
        '--lambda_npl', '10.0',
        '--lambda_recon_l1', '1.0',
        '--lambda_recon_lpips', '1.0',
        '--legacy_freq_weight', '0.0',
        '--save_dir', str(TRAIN_ARTIFACTS),
        '--output_ckpt', str(TRAIN_CHECKPOINT),
        '--noise_prototype', str(TRAIN_NOISE_PROTOTYPE),
        '--log_path', str(TRAIN_INTERNAL_LOG),
    ]


def build_smoke_cmd() -> list[str]:
    return [
        str(PIXART_PYTHON), str(PIXART_ROOT / 'main.py'),
        '--output_mode', 'final',
        '--clean_dataroot', str(CLEAN_DATAROOT),
        '--real_mode', 'symlink',
        '--save_origin', '0',
        '--mode', 'double',
        '--images_root', str(SMOKE35_FAKE_ROOT),
        '--diffusion_steps', '20',
        '--start_step', '18',
        '--iterations', '5',
        '--lr', '1e-3',
        '--res', '224',
        '--dataset_name', 'ours_try',
        '--guidance', '0.0',
        '--eps', str(4 / 255),
        '--attack_loss_weight', '10',
        '--pgd_steps', '10',
        '--lambda_l1', '1.0',
        '--lambda_lpips', '0.5',
        '--lambda_latent', '0.0',
        '--attack_only_fake', '1',
        '--seed', '42',
        '--save_intermediates',
        '--model_name', 'S,E,R,D',
        '--is_encoder', '1',
        '--encoder_weights', str(TRAIN_CHECKPOINT),
        '--pretrained_diffusion_path', str(PIXART_MODEL_PATH),
        '--out_dataroot', str(SMOKE_OUTPUT_DATAROOT),
        '--intermediate_root', str(SMOKE_RUN_DIR / 'intermediates'),
    ]


def prepare_reconstruction_manifest() -> None:
    adv_files = sorted(RECON_SAVE_DIR.glob('*_adv_image.png'))
    if len(adv_files) == 0:
        raise RuntimeError(f'No reconstruction outputs found under {RECON_SAVE_DIR}')
    write_text(RECON_MANIFEST, '\n'.join(str(p) for p in adv_files) + '\n')


def validate_reconstruction_outputs() -> None:
    prepare_reconstruction_manifest()
    rows = [x.strip() for x in RECON_MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(rows) != 700:
        raise ValueError(f'Expected 700 reconstruction images, got {len(rows)}')
    if any(Path(x).is_symlink() for x in rows):
        raise ValueError('Reconstruction manifest contains symlinks; expected real files')


def run_reconstruct() -> None:
    write_suite_spec()
    write_stage_meta(reconstruction_meta(), RECON_META_JSON, RECON_META_MD)
    ensure_dir(RECON_ROOT)
    run_reconstruct_batch()
    validate_reconstruction_outputs()


def run_train() -> None:
    write_suite_spec()
    write_stage_meta(train_meta(), TRAIN_META_JSON, TRAIN_META_MD)
    ensure_dir(TRAIN_ROOT)
    if not RECON_MANIFEST.exists():
        validate_reconstruction_outputs()
    if TRAIN_CHECKPOINT.exists():
        raise FileExistsError(f'Training checkpoint already exists: {TRAIN_CHECKPOINT}')
    run_cmd(build_train_cmd(), cwd=PIXART_ROOT, stdout_path=TRAIN_STDOUT)
    if not TRAIN_CHECKPOINT.exists():
        raise FileNotFoundError(f'Expected checkpoint not found: {TRAIN_CHECKPOINT}')
    if not TRAIN_NOISE_PROTOTYPE.exists():
        raise FileNotFoundError(f'Expected noise prototype not found: {TRAIN_NOISE_PROTOTYPE}')


def export_metrics_for_smoke() -> None:
    surrogate = 'S'
    out_dir = SMOKE_RUN_DIR
    adv_dataroot = SMOKE_OUTPUT_DATAROOT
    export_csv = out_dir / f'adv_logits_sur{surrogate}.csv'
    run_cmd([
        str(PIXART_PYTHON), str(DIFFSR_ROOT / 'evaluation' / 'export_detector_logits.py'),
        '--dataroot', str(adv_dataroot), '--models', 'E,R,D,S', '--batch', '32', '--out_csv', str(export_csv),
    ], cwd=DIFFSR_ROOT, stdout_path=SMOKE_STDOUT)
    run_cmd([
        str(PIXART_PYTHON), str(DIFFSR_ROOT / 'evaluation' / 'transfer_asr_matrix.py'),
        '--run', f'S={export_csv}', '--out_csv', str(out_dir / 'transfer_asr.csv'),
        '--by_subset_csv', str(out_dir / 'transfer_asr_by_subset.csv'),
    ], cwd=DIFFSR_ROOT, stdout_path=SMOKE_STDOUT)
    visual_dir = out_dir / 'metrics' / 'visual'
    run_cmd([
        str(PIXART_PYTHON), str(DIFFSR_ROOT / 'evaluation' / 'eval_visual_quality.py'),
        '--clean_dataroot', str(CLEAN_DATAROOT), '--adv_dataroot', str(adv_dataroot), '--out_dir', str(visual_dir), '--save_per_image',
    ], cwd=DIFFSR_ROOT, stdout_path=SMOKE_STDOUT)
    freq_root = out_dir / '_frequency_root'
    ensure_dir(freq_root)
    ensure_symlink(freq_root / adv_dataroot.name, adv_dataroot)
    run_cmd([
        str(PIXART_PYTHON), str(DIFFSR_ROOT / 'evaluation' / 'eval_frequency_metrics.py'),
        '--clean_dataroot', str(CLEAN_DATAROOT), '--root', str(freq_root), '--out_dir', str(out_dir / 'metrics' / 'frequency'), '--surrogates', 'S', '--res', '224',
    ], cwd=DIFFSR_ROOT, stdout_path=SMOKE_STDOUT)


def collect_smoke_summary() -> dict[str, object]:
    out_dir = SMOKE_RUN_DIR
    summary_metrics = json.loads((SMOKE_OUTPUT_DATAROOT / 'summary_metrics.json').read_text(encoding='utf-8'))
    transfer_row = read_csv_dicts(out_dir / 'transfer_asr.csv')[0]
    visual_row = read_csv_dicts(out_dir / 'metrics' / 'visual' / 'visual_quality.csv')[0]
    freq_rows = read_csv_dicts(out_dir / 'metrics' / 'frequency' / 'frequency_metrics.csv')
    freq_row = next(row for row in freq_rows if row['dataset'] == 'surS')
    summary = {
        'run_id': 'pixart_ctrlvae_pixartproto_surS',
        'backbone': 'pixart',
        'surrogate': 'S',
        'controlvae': 1,
        'adv_dataroot': str(SMOKE_OUTPUT_DATAROOT),
        'encoder_weights': str(TRAIN_CHECKPOINT),
        'whitebox_asr': float(transfer_row['whitebox_asr']),
        'blackbox_mean_asr': float(transfer_row['blackbox_mean_asr']),
        'asr_to_E': float(transfer_row['asr_to_E']) if transfer_row['asr_to_E'] else '',
        'asr_to_R': float(transfer_row['asr_to_R']) if transfer_row['asr_to_R'] else '',
        'asr_to_D': float(transfer_row['asr_to_D']) if transfer_row['asr_to_D'] else '',
        'asr_to_S': '',
        'lpips': float(visual_row['lpips_mean']),
        'psnr': float(visual_row['psnr_mean']),
        'ssim': float(visual_row['ssim_mean']),
        'fourier_l2_to_real': float(freq_row['img_logamp_l2_to_real']),
        'residual_fourier_l2_to_real': float(freq_row['residual_logamp_l2_to_real']),
        'attack_mean_psnr': float(summary_metrics['mean_psnr']),
        'attack_mean_ssim': float(summary_metrics['mean_ssim']),
    }
    write_json(SMOKE_ROOT / 'run_summary.json', summary)
    return summary


def run_smoke_eval() -> None:
    write_suite_spec()
    write_stage_meta(smoke_meta(), SMOKE_META_JSON, SMOKE_META_MD)
    ensure_dir(SMOKE_ROOT)
    ensure_dir(SMOKE_RUN_DIR)
    if not TRAIN_CHECKPOINT.exists():
        raise FileNotFoundError(f'Missing trained checkpoint: {TRAIN_CHECKPOINT}')
    if SMOKE_OUTPUT_DATAROOT.exists():
        raise FileExistsError(f'Smoke output dataroot already exists: {SMOKE_OUTPUT_DATAROOT}')
    run_cmd(build_smoke_cmd(), cwd=PIXART_ROOT, stdout_path=SMOKE_STDOUT)
    export_metrics_for_smoke()
    collect_smoke_summary()


def build_record_candidate() -> dict[str, str]:
    summary = json.loads((SMOKE_ROOT / 'run_summary.json').read_text(encoding='utf-8'))
    row = {k: '' for k in FIELDNAMES}
    row.update({
        'record_id': f'pixart_ctrlvae_pixartproto_smoke35_surS_{EXPERIMENT_DATE}',
        'status': 'complete',
        'project': 'StealthDiffusion_PixArtAlpha',
        'family': 'smoke_fixed35_pixartproto',
        'experiment_name': 'PixArt + ControlVAE retrained with PixArt-specific noise prototype on smoke35 surrogate S',
        'date_hint': EXPERIMENT_DATE,
        'scope': 'smoke_fixed35',
        'data_n': '35',
        'result_path': str(SMOKE_ROOT),
        'backbone': 'pixart',
        'surrogate': 'S',
        'controlvae': '1',
        'iterations': '5',
        't_start': '',
        'lr': '1e-3',
        'lambda_perc': '0.5',
        'primary_change': 'pixart_specific_noise_prototype',
        'whitebox_asr': fmt(summary['whitebox_asr']),
        'blackbox_mean_asr': fmt(summary['blackbox_mean_asr']),
        'asr_to_E': fmt(summary['asr_to_E']),
        'asr_to_R': fmt(summary['asr_to_R']),
        'asr_to_D': fmt(summary['asr_to_D']),
        'asr_to_S': '',
        'lpips': fmt(summary['lpips']),
        'psnr': fmt(summary['psnr']),
        'ssim': fmt(summary['ssim']),
        'fourier_l2_to_real': fmt(summary['fourier_l2_to_real']),
        'residual_fourier_l2_to_real': fmt(summary['residual_fourier_l2_to_real']),
        'attack_mean_psnr': fmt(summary['attack_mean_psnr']),
        'attack_mean_ssim': fmt(summary['attack_mean_ssim']),
        'compared_to': 'pixart_noctrl_700_surS|pixart_ctrl_700_surS',
        'new_information': 'Tests whether replacing the noise prototype with a PixArt-specific one can recover blackbox transfer and frequency closeness under the original ControlVAE training code path.',
        'evidence': str(SMOKE_ROOT / 'run_summary.json'),
    })
    return row


def write_record_candidate() -> None:
    row = build_record_candidate()
    write_json(SMOKE_RECORD_JSON, row)
    lines = ['# Experiment record candidate', '']
    for key in FIELDNAMES:
        lines.append(f'- {key}: {row.get(key, "")}')
    write_text(SMOKE_RECORD_MD, '\n'.join(lines) + '\n')


def append_or_replace_record() -> None:
    row = build_record_candidate()
    with EXPERIMENT_RECORD_CSV.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        header = f.readline()
    target_id = row['record_id']
    existing_index = next((i for i, r in enumerate(rows) if r.get('record_id') == target_id), None)
    if existing_index is None:
        max_order = max(int(r['record_order']) for r in rows if r.get('record_order')) if rows else 0
        row['record_order'] = str(max_order + 1)
        rows.append(row)
    else:
        row['record_order'] = rows[existing_index]['record_order']
        rows[existing_index] = row
    with EXPERIMENT_RECORD_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'reconstruct', 'train', 'smoke_eval', 'record_candidate', 'append_record'])
    args = ap.parse_args()

    if args.action == 'prepare':
        write_suite_spec()
        write_stage_meta(reconstruction_meta(), RECON_META_JSON, RECON_META_MD)
        write_stage_meta(train_meta(), TRAIN_META_JSON, TRAIN_META_MD)
        write_stage_meta(smoke_meta(), SMOKE_META_JSON, SMOKE_META_MD)
        return 0
    if args.action == 'reconstruct':
        run_reconstruct()
        return 0
    if args.action == 'train':
        run_train()
        return 0
    if args.action == 'smoke_eval':
        run_smoke_eval()
        return 0
    if args.action == 'record_candidate':
        write_record_candidate()
        return 0
    append_or_replace_record()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
