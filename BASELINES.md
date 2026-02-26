# Baseline Attacks (整理版)

This repo includes a unified baseline runner:

- `scripts/run_baseline_attack.py`

It is based on the ideas / parameterizations in:

- `/root/gpufree-data/StealthDiffusion_BigGAN/A_downsample_attack.py`
- `/root/gpufree-data/StealthDiffusion_BigGAN/A_FGSM.py`
- `/root/gpufree-data/StealthDiffusion_BigGAN/A_PGD.py`
- `/root/gpufree-data/StealthDiffusion_BigGAN/A_downsample_super.py` (future work; not wired yet)
- `/root/gpufree-data/DiffAttack` (DiffAttack baseline; diffusion-latent attack from TPAMI 2024)

## Output format (compatible with our eval pipeline)

The baseline runner saves:

```
<save_dir>/
  0000_originImage.png
  0000_adv_image.png
  0001_originImage.png
  0001_adv_image.png
  ...
  manifest.csv
  <attack>.log
```

This matches the diffusion attack naming and can be fed into:

- `scripts/build_adv_dataroot.py` to construct `real(clean)+fake(adv)` dataroot
- `scripts/eval_detectors.py` to compute metrics and export per-image logits CSV

## Assumptions / labels

- We run baselines on **fake images only** (label=0).
- Detectors are treated as a 2-class problem:
  - class 0: fake (AI)
  - class 1: real (nature)
- Prediction is computed as `logit > 0` => real(1), else fake(0).

## 1) DownUp resize (black-box transform baseline)

Equivalent idea to `A_downsample_attack.py`:

```bash
python scripts/run_baseline_attack.py \
  --attack downup \
  --images_root exp/genimage_eval_700/fake \
  --save_dir /root/gpufree-data/dataset/baseline_downup_surS_700 \
  --model_name "S,E,R,D" \
  --res 224 \
  --scale_w 0.5 --scale_h 0.75 \
  --down_filter LANCZOS --up_filter BICUBIC
```

## 2) FGSM baseline (white-box on surrogate)

Equivalent idea to `A_FGSM.py`:

```bash
python scripts/run_baseline_attack.py \
  --attack fgsm \
  --images_root exp/genimage_eval_700/fake \
  --save_dir /root/gpufree-data/dataset/baseline_fgsm_surS_700 \
  --model_name "S,E,R,D" \
  --res 224 \
  --eps 0.01568627 \
  --batch 16
```

## 3) PGD baseline (white-box on surrogate)

Equivalent idea to `A_PGD.py`:

```bash
python scripts/run_baseline_attack.py \
  --attack pgd \
  --images_root exp/genimage_eval_700/fake \
  --save_dir /root/gpufree-data/dataset/baseline_pgd_surS_700 \
  --model_name "S,E,R,D" \
  --res 224 \
  --eps 0.01568627 \
  --pgd_alpha 0.00392157 \
  --pgd_steps 10 \
  --batch 16
```

## Evaluate baselines (post-attack)

Build a dataroot:

```bash
python scripts/build_adv_dataroot.py \
  --clean_dataroot exp/genimage_eval_700 \
  --attack_out /root/gpufree-data/dataset/baseline_pgd_surS_700 \
  --out_dataroot exp/genimage_eval_700_adv_pgd_surS \
  --mode symlink
```

Evaluate and export per-image logits:

```bash
python scripts/eval_detectors.py \
  --dataroot exp/genimage_eval_700_adv_pgd_surS \
  --models E,R,D,S \
  --batch 32 \
  --out_csv exp/results/adv_logits_pgd_surS.csv
```

## 4) DiffAttack baseline (diffusion-latent attack)

We provide a wrapper that runs DiffAttack code on our dataset while using our detectors (E/R/D/S) as surrogate.

Generate adversarial fakes (surrogate=S by default):

```bash
python scripts/run_diffattack_baseline.py \
  --images_root exp/genimage_eval_700/fake \
  --save_dir /root/gpufree-data/dataset/baseline_diffattack_surS_700 \
  --model_name S \
  --pretrained_diffusion_path stabilityai/stable-diffusion-2-1-base \
  --res 224 \
  --diffusion_steps 20 --start_step 18 --iterations 5 \
  --guidance 2.5 \
  --attack_loss_weight 10 \
  --cross_attn_loss_weight 10000 \
  --self_attn_loss_weight 100
```

Then reuse the same post-attack mapping + eval pipeline:

```bash
python scripts/build_adv_dataroot.py \
  --clean_dataroot exp/genimage_eval_700 \
  --attack_out /root/gpufree-data/dataset/baseline_diffattack_surS_700 \
  --out_dataroot exp/genimage_eval_700_adv_diffattack_surS \
  --mode symlink

python scripts/eval_detectors.py \
  --dataroot exp/genimage_eval_700_adv_diffattack_surS \
  --models E,R,D,S \
  --batch 32 \
  --out_csv exp/results/adv_logits_diffattack_surS.csv
```
