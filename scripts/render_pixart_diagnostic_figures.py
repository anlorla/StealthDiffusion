#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

REPO_ROOT = Path('/root/gpufree-data')
CHECKPOINT_ROOT = REPO_ROOT / 'checkpoints'
RESULT_ROOT = REPO_ROOT / 'StealthDiffusion_PixArtAlpha' / 'results' / 'pixart_ctrlvae_pixartproto_20260324'
DIAG_ROOT = RESULT_ROOT / 'diagnostics'
RECON_ROOT = DIAG_ROOT / 'recon_only_smoke35'
PROTO_ROOT = DIAG_ROOT / 'prototype_compare'
NEW_PROTO_PATH = CHECKPOINT_ROOT / 'pixart_noise_prototype_20260324.pt'
OLD_PROTO_PATH = PROTO_ROOT / 'oldstyle_cleanreal700_noise_prototype.pt'
OUT_ROOT = DIAG_ROOT / 'publication_figures'

plt.rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'DejaVu Sans',
})


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def save_fig(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUT_ROOT)
    png = OUT_ROOT / f'{stem}.png'
    pdf = OUT_ROOT / f'{stem}.pdf'
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {png}')
    print(f'saved {pdf}')


def load_proto(path: Path) -> np.ndarray:
    tensor = torch.load(path, map_location='cpu').float().squeeze(0)
    return tensor.numpy()


def rgb_composite(arr_chw: np.ndarray, vmax: float) -> np.ndarray:
    denom = vmax if vmax > 0 else 1.0
    norm = np.clip(arr_chw / denom, -1.0, 1.0)
    return np.transpose((norm * 0.5 + 0.5), (1, 2, 0))


def gray_signed(arr_chw: np.ndarray) -> np.ndarray:
    return np.mean(arr_chw, axis=0)


def fft_logamp(gray_hw: np.ndarray) -> np.ndarray:
    fft = np.fft.fftshift(np.fft.fft2(gray_hw))
    return np.log1p(np.abs(fft)).astype(np.float32)


def radial_profile(arr_hw: np.ndarray) -> np.ndarray:
    h, w = arr_hw.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = int(rr.max())
    values = []
    for radius in range(max_r + 1):
        mask = rr == radius
        values.append(float(arr_hw[mask].mean()) if mask.any() else np.nan)
    return np.asarray(values, dtype=np.float32)


def render_recon_only(summary: dict) -> None:
    modes = ['pixart_vae', 'ctrlvae_old', 'ctrlvae_new']
    labels = ['PixArt VAE', 'Old ControlVAE', 'New PixArt-specific ControlVAE']
    colors = ['#4C78A8', '#54A24B', '#E45756']
    metrics = [
        ('psnr_mean', 'PSNR ↑', '{:.2f}'),
        ('ssim_mean', 'SSIM ↑', '{:.3f}'),
        ('fourier_l2_pair_mean', 'Fourier L2 to input ↓', '{:.1f}'),
        ('dataset_mean_logamp_l2_to_real700', 'Dataset log-amp L2 to real700 ↓', '{:.1f}'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title, fmt) in zip(axes, metrics):
        vals = [summary[m][key] for m in modes]
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', rotation=12)
        ymin = min(vals)
        ymax = max(vals)
        pad = (ymax - ymin) * 0.18 if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad * 0.15, ymax + pad)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad * 0.08, fmt.format(v), ha='center', va='bottom')
    fig.suptitle('Reconstruction-only diagnostic on smoke35 (no attack, no LAO)', fontsize=15, y=1.01)
    fig.text(0.5, 0.01, 'Interpretation target: if the new ControlVAE is already weak here, the issue is in prototype training/objective rather than LAO interaction.', ha='center', fontsize=10)
    save_fig(fig, 'recon_only_metrics_publication')


def render_proto_spatial(old_proto: np.ndarray, new_proto: np.ndarray) -> None:
    old_gray = gray_signed(old_proto)
    new_gray = gray_signed(new_proto)
    diff_gray = new_gray - old_gray
    shared_gray_max = max(np.abs(old_gray).max(), np.abs(new_gray).max(), 1e-8)
    diff_max = max(np.abs(diff_gray).max(), 1e-8)
    rgb_max = max(np.abs(old_proto).max(), np.abs(new_proto).max(), 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
    rgb_panels = [
        ('Old-style prototype (RGB residual composite)', rgb_composite(old_proto, rgb_max)),
        ('New PixArt prototype (RGB residual composite)', rgb_composite(new_proto, rgb_max)),
        ('Difference: new - old (RGB residual composite)', rgb_composite(new_proto - old_proto, max(np.abs(new_proto - old_proto).max(), 1e-8))),
    ]
    for ax, (title, img) in zip(axes[0], rgb_panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    norm_main = TwoSlopeNorm(vmin=-shared_gray_max, vcenter=0.0, vmax=shared_gray_max)
    norm_diff = TwoSlopeNorm(vmin=-diff_max, vcenter=0.0, vmax=diff_max)
    heat_panels = [
        ('Old-style prototype (channel-mean signed map)', old_gray, norm_main),
        ('New PixArt prototype (channel-mean signed map)', new_gray, norm_main),
        ('Difference: new - old', diff_gray, norm_diff),
    ]
    images = []
    for ax, (title, arr, norm) in zip(axes[1], heat_panels):
        im = ax.imshow(arr, cmap='coolwarm', norm=norm)
        images.append(im)
        ax.set_title(title)
        ax.axis('off')
    cbar1 = fig.colorbar(images[0], ax=axes[1, :2], shrink=0.85, location='bottom', pad=0.06)
    cbar1.set_label('Signed residual intensity (shared scale across old/new)')
    cbar2 = fig.colorbar(images[2], ax=axes[1, 2], shrink=0.85, location='bottom', pad=0.06)
    cbar2.set_label('Difference intensity')
    fig.suptitle('Spatial-domain prototype comparison', fontsize=15, y=1.01)
    fig.text(0.5, 0.02, 'Top row keeps RGB residual color cues. Bottom row uses zero-centered heatmaps to reveal sign structure and boundary artifacts.', ha='center', fontsize=10)
    save_fig(fig, 'prototype_spatial_publication')


def render_proto_fft(old_proto: np.ndarray, new_proto: np.ndarray) -> None:
    old_fft = fft_logamp(gray_signed(old_proto))
    new_fft = fft_logamp(gray_signed(new_proto))
    diff_fft = new_fft - old_fft
    shared_min = min(old_fft.min(), new_fft.min())
    shared_max = max(old_fft.max(), new_fft.max())
    diff_abs = max(np.abs(diff_fft).max(), 1e-8)
    h, w = old_fft.shape
    cy, cx = h // 2, w // 2
    radius = 32
    slices = np.s_[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), constrained_layout=True)
    panels = [
        ('Old-style FFT log-amplitude', old_fft, 'magma', (shared_min, shared_max)),
        ('New PixArt FFT log-amplitude', new_fft, 'magma', (shared_min, shared_max)),
        ('Difference: new - old', diff_fft, 'coolwarm', (-diff_abs, diff_abs)),
    ]
    ims_top = []
    ims_bottom = []
    for ax, (title, arr, cmap, lims) in zip(axes[0], panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lims[0], vmax=lims[1])
        ims_top.append(im)
        ax.set_title(title)
        ax.axis('off')
        rect = plt.Rectangle((cx - radius, cy - radius), 2 * radius + 1, 2 * radius + 1, edgecolor='cyan', facecolor='none', linewidth=1.2)
        ax.add_patch(rect)
    for ax, (title, arr, cmap, lims) in zip(axes[1], panels):
        im = ax.imshow(arr[slices], cmap=cmap, vmin=lims[0], vmax=lims[1])
        ims_bottom.append(im)
        ax.set_title(title + ' (center zoom)')
        ax.axis('off')
    cbar1 = fig.colorbar(ims_top[0], ax=axes[:, :2], shrink=0.82, location='bottom', pad=0.07)
    cbar1.set_label('Log amplitude (shared across old/new)')
    cbar2 = fig.colorbar(ims_top[2], ax=axes[:, 2], shrink=0.82, location='bottom', pad=0.07)
    cbar2.set_label('Log-amplitude difference')
    fig.suptitle('Frequency-domain prototype comparison', fontsize=15, y=1.01)
    fig.text(0.5, 0.02, 'Top row shows the full spectrum; bottom row zooms into the center to inspect low-frequency peak shape and axis-aligned structure.', ha='center', fontsize=10)
    save_fig(fig, 'prototype_fft_publication')


def render_proto_radial_and_stats(report: dict) -> None:
    old = report['oldstyle_clean_real700']
    new = report['new_pixart_proto']
    ratios = report['new_over_old']
    old_rad = np.asarray(old['radial_profile'], dtype=np.float32)
    new_rad = np.asarray(new['radial_profile'], dtype=np.float32)
    radii = np.arange(min(len(old_rad), len(new_rad)))

    metric_names = ['L2 norm', 'Mean abs', 'FFT energy', 'Center peak ratio']
    old_vals = [old['l2_norm'], old['mean_abs'], old['fft_energy_mean'], old['center_peak_overall_ratio']]
    new_vals = [new['l2_norm'], new['mean_abs'], new['fft_energy_mean'], new['center_peak_overall_ratio']]
    ratio_vals = [ratios['l2_norm_ratio'], ratios['mean_abs_ratio'], ratios['fft_energy_ratio'], ratios['center_peak_ratio_ratio']]

    fig = plt.figure(figsize=(14, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.plot(radii, old_rad[:len(radii)], label='Old-style prototype', color='#4C78A8', linewidth=2.2)
    ax0.plot(radii, new_rad[:len(radii)], label='New PixArt prototype', color='#E45756', linewidth=2.2)
    ax0.set_title('Radial log-amplitude profile')
    ax0.set_xlabel('Radius from FFT center')
    ax0.set_ylabel('Mean log amplitude')
    ax0.grid(True, linestyle='--', alpha=0.28)
    ax0.legend(frameon=False)

    x = np.arange(len(metric_names))
    width = 0.36
    ax1.bar(x - width / 2, old_vals, width=width, color='#4C78A8', label='Old-style')
    ax1.bar(x + width / 2, new_vals, width=width, color='#E45756', label='New PixArt')
    ax1.set_yscale('log')
    ax1.set_title('Absolute prototype statistics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=18, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.28)
    ax1.legend(frameon=False)

    bars = ax2.bar(np.arange(len(metric_names)), ratio_vals, color=['#F58518', '#F58518', '#B279A2', '#72B7B2'])
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax2.set_yscale('log')
    ax2.set_title('New / old ratios')
    ax2.set_xticks(np.arange(len(metric_names)))
    ax2.set_xticklabels(metric_names, rotation=18, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.28)
    for bar, value in zip(bars, ratio_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f'{value:.4f}x', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Prototype scale and frequency-shape summary', fontsize=15, y=1.01)
    fig.text(0.5, 0.01, 'If the new prototype were over-aggressive, these ratios would exceed 1. Instead, all major magnitude indicators are far below 1.', ha='center', fontsize=10)
    save_fig(fig, 'prototype_radial_stats_publication')


def main() -> int:
    ensure_dir(OUT_ROOT)
    recon_summary = load_json(RECON_ROOT / 'summary.json')
    proto_report = load_json(PROTO_ROOT / 'prototype_compare_report.json')
    old_proto = load_proto(OLD_PROTO_PATH)
    new_proto = load_proto(NEW_PROTO_PATH)

    render_recon_only(recon_summary)
    render_proto_spatial(old_proto, new_proto)
    render_proto_fft(old_proto, new_proto)
    render_proto_radial_and_stats(proto_report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
