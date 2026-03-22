# PixArt-Alpha worklog (2026-03-22)

## Goal

Start the backbone migration from SD v2.1 / UNet toward PixArt-Alpha / Transformer in an isolated copy of StealthDiffusion, and get the codebase to a state where PixArt-Alpha diagnostics can run as soon as a local checkpoint is available.

## Workspace boundary

- Working repo: `/root/gpufree-data/StealthDiffusion_PixArtAlpha`
- Reference only: `/root/gpufree-data/StealthDiffusion`
- Not modified in this round: `/root/gpufree-data/StealthDiffusion_SR`

## Problem / pipeline / metrics / IO

### Problem

Verify whether the StealthDiffusion latent-attack backbone can be generalized from SD v2.1 to PixArt-Alpha.

### Pipeline

Current target pipeline for this stage:

1. load diffusion backbone
2. encode image to VAE latent
3. build text condition / CFG context
4. run DDIM inversion or DDIM denoising through the denoiser backbone
5. decode latent with VAE
6. compare reconstruction metrics

Candidate backbone modules now considered in this repo:

- `StableDiffusionPipeline`
- `PixArtAlphaPipeline`

### Metrics for this stage

Diagnostics only:

- native pipeline health check
- VAE reconstruction PSNR / SSIM
- DDIM inversion + decode PSNR / SSIM
- single forward/backward memory probe

### Input data

- clean test image used in this round:
  `/root/gpufree-data/dataset/original_genimage_eval_1400/fake/adm/385_adm_145.PNG`
- local SD checkpoint used as baseline:
  `/root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06`
- intended PixArt-Alpha checkpoint path:
  `/root/gpufree-data/checkpoints/hf_models/pixart-alpha-xl-2-512x512`

### Output locations

- diagnostic outputs:
  `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked`
- this worklog:
  `/root/gpufree-data/StealthDiffusion_PixArtAlpha/PIXART_ALPHA_WORKLOG_20260322.md`

### Weights

- no new weights were trained in this round
- no ControlVAE retraining was performed

## Code changes made

### 1. `main.py`

Added backbone-aware pipeline loading.

Key changes:

- added `_infer_pipeline_class()`
- added `load_diffusion_pipeline()`
- supports two backbone families:
  - `StableDiffusionPipeline`
  - `PixArtAlphaPipeline`
- normalizes scheduler setup to DDIM + DDIMInverseScheduler for future inversion diagnostics

Quick local probe result:

- `_infer_pipeline_class('/tmp/pixart-alpha-model') -> PixArtAlphaPipeline`
- `_infer_pipeline_class('<local sd path>') -> StableDiffusionPipeline`

### 2. `diff_latent_attack.py`

Refactored the attack-side latent utilities so the denoiser is no longer hard-coded to `model.unet`.

Added backbone helpers:

- `is_pixart_alpha_backbone()`
- `get_denoiser()`
- `get_scaling_factor()`
- `encode_prompt_embeddings()`
- `build_added_cond_kwargs()`
- `build_context()`
- `get_model_prediction()`

Updated existing paths to use these helpers in:

- latent encoding
- DDIM inversion
- latent initialization
- DDIM denoising step
- VAE decoding scale handling
- denoiser `requires_grad_(False)` path

Important compatibility point:

- PixArt-Alpha uses `transformer(...)` + `encoder_attention_mask` instead of `unet(...)` + `encoder_hidden_states` only.
- the new wrapper follows the official diffusers PixArt-Alpha denoising call pattern, including the learned-sigma output split.

### 3. New diagnostic script

Added:

- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/scripts/diagnose_pixart_alpha.py`

What it does:

- records environment / checkpoint / network status
- if SD local assets exist, runs SD baseline recon diagnostics immediately
- if PixArt-Alpha local checkpoint exists, it is ready to run:
  - native PixArt generation
  - PixArt VAE recon
  - PixArt DDIM recon
  - PixArt memory probe
- if PixArt checkpoint is missing, it writes a blocker note instead of crashing

## Validation run completed

I ran:

```bash
cd /root/gpufree-data/StealthDiffusion_PixArtAlpha
/opt/conda/envs/sd/bin/python scripts/diagnose_pixart_alpha.py \
  --out_dir /root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked
```

Generated files:

- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked/env_summary.json`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked/summary.json`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked/note.md`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked/input_resized.png`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked/sd21_vae_recon.png`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_blocked/sd21_ddim_recon.png`

## Results obtained in this round

From `summary.json`:

- SD VAE recon on the chosen 512x512 test image:
  - PSNR = `25.758996176816176`
  - SSIM = `0.7708166702137831`
- SD DDIM inversion + decode on the same image:
  - PSNR = `21.57295915471288`
  - SSIM = `0.6221798758622926`

These are not the final target PixArt numbers, but they give us a local reference point on the exact same image and exact same diagnostic script.

## Blocker encountered

PixArt-Alpha could not be run yet because both of these were true during execution:

- local checkpoint missing at:
  `/root/gpufree-data/checkpoints/hf_models/pixart-alpha-xl-2-512x512`
- network probe to `huggingface.co:443` failed during this session

Recorded environment facts:

- Python env: `/opt/conda/envs/sd/bin/python`
- GPU: `NVIDIA GeForce RTX 4090`
- `PixArtAlphaPipeline` import is available in local `diffusers`
- `PixArtSigmaPipeline` is not available in the current `diffusers==0.24.0` env

## What is ready now

As soon as a local PixArt-Alpha checkpoint is placed under the intended path, this repo is ready for the next round:

1. rerun `scripts/diagnose_pixart_alpha.py`
2. obtain native PixArt health-check output
3. obtain PixArt VAE reconstruction metrics
4. test PixArt DDIM inversion path
5. if those pass, continue into attack-side smoke experiments

## Files changed in this round

- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/main.py`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/diff_latent_attack.py`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/scripts/diagnose_pixart_alpha.py`
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/PIXART_ALPHA_WORKLOG_20260322.md`

## Notes on repository state

This clone inherited some pre-existing uncommitted changes from the source StealthDiffusion working tree. I did not revert them. My new work in this round is limited to the files listed above plus the generated diagnostics under `results/pixart_alpha_diagnostics/`.
