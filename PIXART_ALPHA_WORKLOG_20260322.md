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

## DDIM diagnostic fixes locked in (2026-03-22)

These are the settings that fixed the previously low PixArt DDIM reconstruction numbers during diagnostics.

### 1. Deterministic VAE encode for diagnostics

- code path:
  `/root/gpufree-data/StealthDiffusion_PixArtAlpha/diff_latent_attack.py`
- setting:
  `sample_posterior=False`
- meaning:
  use the VAE posterior mode instead of sampling from the posterior
- why it matters:
  removes VAE sampling noise from diagnostics so the DDIM reconstruction score reflects scheduler / backbone behavior rather than stochastic encode noise

### 2. PixArt local loading mode

- code path:
  `/root/gpufree-data/StealthDiffusion_PixArtAlpha/main.py`
- settings:
  - `variant="fp16"`
  - `use_safetensors=True`
- meaning:
  force diffusers to load the local fp16 PixArt checkpoint files that were downloaded from Hugging Face
- why it matters:
  avoids loader mismatch against missing `.bin` weights and ensures the intended local checkpoint is used

### 3. PixArt DDIM scheduler conversion fix

- code path:
  `/root/gpufree-data/StealthDiffusion_PixArtAlpha/main.py`
- effective settings for PixArt when converting the original scheduler into DDIM:
  - `prediction_type="epsilon"` (kept as-is from the checkpoint)
  - `timestep_spacing="trailing"`
  - `clip_sample=False`
  - `set_alpha_to_one=False`
  - `steps_offset=1`
- meaning of each field:
  - `prediction_type`:
    tells the scheduler how to interpret the model output; here the PixArt checkpoint is treated as predicting noise `epsilon`
  - `timestep_spacing`:
    controls which discrete timesteps are selected from the 1000-step training grid; `trailing` keeps the inverse scheduler compatible and gives stable reconstruction
  - `clip_sample`:
    whether to clamp predicted `x0` during DDIM stepping; disabling clipping preserves latent values and avoids reconstruction damage
  - `set_alpha_to_one`:
    controls the alpha used at the boundary step; setting it to `False` matches latent-diffusion-style DDIM behavior better here
  - `steps_offset`:
    shifts the timestep grid by one; `1` aligns the DDIM grid better with the SD-style latent setup used in this repo
- why it matters:
  the previous DDIM conversion effectively fell back to unsuitable defaults (`clip_sample=True`, `set_alpha_to_one=True`, `steps_offset=0`), which was the main reason PixArt DDIM reconstruction was stuck around `19 dB`

### 4. PixArt inverse scheduler pairing

- code path:
  `/root/gpufree-data/StealthDiffusion_PixArtAlpha/main.py`
- setting:
  `pipe.inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)`
- meaning:
  inversion and denoising now use matched DDIM / DDIMInverse configurations derived from the same corrected config
- why it matters:
  keeps the reverse and forward processes aligned during the reconstruction diagnostic

### 5. Diagnostic sweep protocol

- code path:
  `/root/gpufree-data/StealthDiffusion_PixArtAlpha/scripts/diagnose_pixart_alpha.py`
- settings:
  - single-image diagnostics use deterministic encode
  - DDIM sweep steps: `20,50,100`
- meaning:
  evaluate whether reconstruction improves as the inversion / denoising grid becomes finer
- why it matters:
  confirms that the fixed scheduler configuration behaves sensibly across multiple step counts

### Result after these fixes

- PixArt DDIM reconstruction improved from about `18.84 / 19.04 / 19.10 dB` at `20 / 50 / 100` steps
- to about `23.21 / 23.45 / 24.55 dB`
- this indicates the main blocker in diagnostic 3 was scheduler-conversion configuration, not a fundamental PixArt backbone failure

## Memory / gradient-checkpointing follow-up (2026-03-22)

### Code changes

- `diff_latent_attack.py`
  - added `move_pipeline_modules_to_cpu()`
  - added `offload_unused_conditioning_modules()`
  - after building attack contexts, the code now offloads unused conditioning modules such as `text_encoder`
- `main.py`
  - added CLI flags:
    - `--backbone_gradient_checkpointing`
    - `--backbone_gc_vae`
- `scripts/diagnose_pixart_alpha.py`
  - memory probe now runs two cases:
    - `frozen_no_gc`
    - `denoiser_gc`
  - the probe first precomputes prompt embeddings, then moves `text_encoder` and `vae` to CPU before measuring the denoiser-only backward pass

### Why this change was needed

- the earlier OOM was dominated by full-pipeline GPU residency, not by the single denoiser backward alone
- PixArt keeps a large T5 text encoder on GPU by default; for the latent attack loop, text embeddings are fixed after context construction, so keeping the text encoder resident wastes memory

### New probe result

From:

- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/pixart_alpha_diagnostics/run_20260322_alpha_gcprobe/summary.json`

The denoiser-only probe is now healthy without gradient checkpointing once unused modules are offloaded:

- `frozen_no_gc`
  - moved to CPU: `text_encoder`, `vae`
  - allocated before probe: about `1181.93 MB`
  - peak allocated during forward/backward: about `3005.08 MB`
  - peak reserved: about `3304.00 MB`

### Current limitation of PixArt GC in this env

- the `denoiser_gc` probe still fails in the current `diffusers==0.24.0` style environment with:
  - `Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0`
- so the practical memory fix for now is:
  - offload unused conditioning modules after prompt/context construction
  - keep denoiser frozen
  - treat backbone GC as optional / experimental rather than required

### Practical conclusion

- diagnostic 4 is no longer blocked by a blanket OOM
- for the next alpha smoke run, the first memory optimization to rely on is text-encoder offload, not gradient checkpointing
