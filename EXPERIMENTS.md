# Experiments: GenImage 700 (7x100) Protocol

This repo currently uses:
- `label=0` for `fake` (AI) images
- `label=1` for `real` (genuine) images

We evaluate **detection evasion** by crafting adversarial examples for **fake** images and then measuring how much detector performance drops on the mixed set:

`real(clean) + fake(adv)`

## What “map adv fake back to original structure” means

`main.py` saves adversarial fakes as flat files like `0000_adv_image.png`, `0001_adv_image.png`, ... in an output folder.

But our evaluation expects a dataset “dataroot” layout:

```
dataroot/
  real/...
  fake/<subset>/...
```

So we **rebuild** a new dataroot where:
- `real/` is copied/linked from the clean dataset (unchanged)
- `fake/` contains the **adv images** placed under the same relative fake path order (so subset stats remain correct)

That rebuilt dataroot is what we evaluate post-attack.

## CSV outputs (what they contain and how they’re computed)

`scripts/eval_detectors.py --out_csv ...` writes **one row per image** with:
- `path`: image path
- `label`: 0=fake, 1=real (inferred from the path containing `fake`/`real`)
- `subset`: for fake images, inferred from `fake/<subset>/...` (or best-effort from filename); for real images it’s `real`
- `logit_E`, `logit_R`, `logit_D`, `logit_S`: detector outputs (one per model)

Important details:
- Each detector returns a 2-logit vector `[0, z]` (see `other_attacks.OURS`). We store `z` as `logit_*`.
- Predicted label is `1` (real) if `z > 0`, else `0` (fake). This matches `argmax([0, z])`.
- Overall metrics are computed from `(label, logit)`:
  - `acc`: mean of correct predictions
  - `bacc`: (TPR + TNR)/2
  - `auc`: ROC-AUC using `logit` as the score
  - `ap`: average precision using `logit` as the score
  - `tpr@fpr1%/5%`: maximum TPR achievable while keeping FPR <= 1% / 5%
- For fake-only per-subset reporting, we print:
  - `asr(fake->real) = mean(logit > 0)` over fake images in that subset

## 0) Dataset preparation (already done for fake_100x7)

Count and sample fake subsets (7 subsets x 100 = 700):

```bash
python scripts/prepare_genimage_subsets.py \
  --fake_dir /root/gpufree-data/dataset/test/genimage/fake \
  --out_dir exp/genimage_prepared_v2 \
  --k 100 --n_sets 7 --seed 42 --mode symlink
```

Create a balanced eval dataroot with 700 real + 700 fake:

```bash
python scripts/prepare_genimage_eval.py \
  --fake_sample_dir exp/genimage_prepared_v2/fake_100x7 \
  --real_dir /root/gpufree-data/dataset/test/genimage/real \
  --real_k 700 --seed 42 \
  --out_dir exp/genimage_eval_700 \
  --mode symlink
```

## 1) Clean baseline evaluation (no attack)

```bash
python scripts/eval_detectors.py \
  --dataroot exp/genimage_eval_700 \
  --models E,R,D,S \
  --batch 32 \
  --out_csv exp/results/clean_logits.csv
```

This prints overall metrics and per-fake-subset ASR-like stats (on clean fakes).

## 2) Paper-style attack baseline (example: S as surrogate)

Key paper-ish settings:
- resize `224x224` (`--res 224`)
- PGD init: `eps=4/255`, `steps=10`
- coefficients: alpha=1, beta=1, gamma=10
- latent diffusion: 2 steps + 5 iterations (configured via `diffusion_steps - start_step = 2`)
- surrogate first in `--model_name` (e.g. `S,E,R,D`)

Run attack (craft adv for fake only):

```bash
python main.py \
  --images_root exp/genimage_eval_700/fake \
  --save_dir /root/gpufree-data/dataset/genimage_adversial_surS_700 \
  --attack_only_fake 1 \
  --model_name "S,E,R,D" \
  --res 224 \
  --eps 0.01568627 --pgd_steps 10 \
  --attack_loss_weight 10 --lambda_l1 1 --lambda_lpips 1 --lambda_latent 0 \
  --diffusion_steps 20 --start_step 18 \
  --iterations 5 \
  --is_encoder 1
```

## 3) Build post-attack dataroot: real(clean) + fake(adv)

This step “maps” `0000_adv_image.png` back under `fake/<subset>/...` to preserve subset structure and allow evaluation.

```bash
python scripts/build_adv_dataroot.py \
  --clean_dataroot exp/genimage_eval_700 \
  --attack_out /root/gpufree-data/dataset/genimage_adversial_surR_700 \
  --out_dataroot exp/genimage_eval_700_adv_surR \
  --mode symlink
```

## 4) Post-attack evaluation

```bash
python scripts/eval_detectors.py \
  --dataroot exp/genimage_eval_700_adv_surR \
  --models E,R,D,S \
  --batch 32 \
  --out_csv exp/results/adv_logits_surR.csv
```

Compare `exp/results/clean_logits.csv` vs `exp/results/adv_logits_surS.csv`.

## 5) Full 4x3 transfer ASR matrix (requires 4 runs)

Repeat steps (2)-(4) for each surrogate:
- surrogate E: `--model_name "E,R,D,S"` -> `adv_logits_surE.csv`
- surrogate R: `--model_name "R,E,D,S"` -> `adv_logits_surR.csv`
- surrogate D: `--model_name "D,E,R,S"` -> `adv_logits_surD.csv`
- surrogate S: `--model_name "S,E,R,D"` -> `adv_logits_surS.csv`

Then compute the transfer ASR matrix (12 values = 4*3 non-diagonal entries):

```bash
python scripts/transfer_asr_matrix.py \
  --run E=exp/results/adv_logits_surE.csv \
  --run R=exp/results/adv_logits_surR.csv \
  --run D=exp/results/adv_logits_surD.csv \
  --run S=exp/results/adv_logits_surS.csv \
  --out_csv exp/results/transfer_asr.csv \
  --by_subset_csv exp/results/transfer_asr_by_subset.csv
```

## Notes / gotchas

- GPU: evaluation on CPU can be very slow; if CUDA is available, the scripts will use it automatically.
- Ordering: `build_adv_dataroot.py` assumes the attack output indices `0000,0001,...` match `natsorted(clean_dataroot/fake/**)`. So always run `main.py` on the same `clean_dataroot/fake` you later pass to `build_adv_dataroot.py`.

