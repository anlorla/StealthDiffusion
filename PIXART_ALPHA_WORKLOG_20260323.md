# PixArt-Alpha worklog (2026-03-23)

# PixArt-Alpha full-700 suite spec

## Problem
- Evaluate whether PixArt-Alpha can serve as the no-ControlVAE attack backbone for full GenImage-700 fake-image evasion across surrogates E/R/D/S while preserving visual and frequency stealth.

## Pipeline
- load PixArt-Alpha backbone
- encode clean fake image into latent
- run DDIM inversion
- apply PGD warm start + latent optimization
- run DDIM denoising and decode adversarial image
- export full evaluation dataroot with clean real + adversarial fake
- evaluate detectors, visual quality, and frequency statistics

## Candidate Modules
- enabled: PixArtAlphaPipeline.transformer, vae, tokenizer, text_encoder, detectors E/R/D/S
- disabled: ControlVAE NewEncoder, ControlVAE NewDecoder

## Metrics
- whitebox_asr
- blackbox_mean_asr
- asr_to_E
- asr_to_R
- asr_to_D
- asr_to_S
- lpips
- psnr
- ssim
- fourier_l2_to_real
- residual_fourier_l2_to_real
- attack_mean_psnr
- attack_mean_ssim

## IO
- input clean dataroot: `/root/gpufree-data/dataset/original_genimage_eval_1400`
- input fake source: `/root/gpufree-data/dataset/original_genimage_eval_1400/fake`
- output datasets: `/root/gpufree-data/dataset/pixart_alpha/attack_datasets/full_700`
- output evaluation: `/root/gpufree-data/DiffSRAttack/results/PixArtAlpha_noctrlvae`
- output logs: `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/full_700_20260323`
- output weights: none

# PixArt-Alpha full GenImage-700 suite (2026-03-23)

## Problem
- Evaluate the full 700-fake PixArt-Alpha no-ControlVAE attack matrix across surrogates E/R/D/S.

## Outputs
- Suite root: `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/full_700_20260323`
- Dataset root: `/root/gpufree-data/dataset/pixart_alpha/attack_datasets/full_700`
- Batch eval root: `/root/gpufree-data/DiffSRAttack/results/PixArtAlpha_noctrlvae`

## Best rows
- Whitebox ASR: `pixart_noctrl_700_surS` = `0.941429`
- Blackbox mean ASR: `pixart_noctrl_700_surE` = `0.801429`
- PSNR: `pixart_noctrl_700_surS` = `25.096844`
- Fourier L2 to real: `pixart_noctrl_700_surR` = `27.727842`

## Summary CSV
- `/root/gpufree-data/StealthDiffusion_PixArtAlpha/results/full_700_20260323/suite_summary.csv`
