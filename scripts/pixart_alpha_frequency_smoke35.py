#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path("/root/gpufree-data")
PIXART_ROOT = REPO_ROOT / "StealthDiffusion_PixArtAlpha"
DIFFSR_ROOT = REPO_ROOT / "DiffSRAttack"
DATASET_ROOT = REPO_ROOT / "dataset"
CLEAN_DATAROOT = DATASET_ROOT / "original_genimage_eval_1400"
SMOKE35_FAKE_ROOT = DATASET_ROOT / "smoke_fixed35" / "fake"
ATTACK_DATASETS_ROOT = DATASET_ROOT / "pixart_alpha" / "attack_datasets"
PIXART_PYTHON = REPO_ROOT / "conda_envs" / "pixart_alpha" / "bin" / "python"
PIXART_MODEL_PATH = REPO_ROOT / "checkpoints" / "hf_models" / "pixart-alpha-xl-2-512x512"
PSEUDO_LABEL_CSV = DIFFSR_ROOT / "metadata" / "smoke35_pseudo_labels.csv"
RESULT_ROOT = PIXART_ROOT / "results" / "step3b_frequency_smoke35_20260327"
RUN_ID = "pixart_frequency_loss_surS"
RUN_DIR = RESULT_ROOT / "runs" / RUN_ID
RUN_SUMMARY = RUN_DIR / "run_summary.json"
OUTPUT_DATAROOT = ATTACK_DATASETS_ROOT / "smoke35" / "PixArtAlpha_smoke35_adv_frequency_loss_surS_20260327"

ATTACK_ARGS = {
    "output_mode": "final",
    "clean_dataroot": str(CLEAN_DATAROOT),
    "real_mode": "symlink",
    "save_origin": 0,
    "mode": "double",
    "images_root": str(SMOKE35_FAKE_ROOT),
    "diffusion_steps": 20,
    "start_step": 18,
    "iterations": 5,
    "lr": 1e-3,
    "res": 224,
    "dataset_name": "ours_try",
    "guidance": 0.0,
    "eps": 4 / 255,
    "attack_loss_weight": 10,
    "pgd_steps": 10,
    "lambda_l1": 1.0,
    "lambda_lpips": 0.5,
    "lambda_latent": 0.0,
    "lambda_f": 0.1,
    "attack_only_fake": 1,
    "seed": 42,
    "save_intermediates": True,
    "pseudo_label_csv": str(PSEUDO_LABEL_CSV),
    "model_name": "S,E,R,D",
    "is_encoder": 0,
    "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
    "out_dataroot": str(OUTPUT_DATAROOT),
    "intermediate_root": str(RUN_DIR / "intermediates"),
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, start=link_path.parent)
    os.symlink(rel, link_path)


def copy_text_like(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def attack_log_path() -> Path:
    return PIXART_ROOT / "exp" / "results" / "attack_SERD0_surS.log"


def run_attack() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ATTACK_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DATAROOT.exists():
        raise FileExistsError(f"Output dataroot already exists: {OUTPUT_DATAROOT}")

    log_path = attack_log_path()
    start_size = log_path.stat().st_size if log_path.exists() else 0
    write_json(
        RUN_DIR / "run_meta.json",
        {
            "run_id": RUN_ID,
            "problem": "Measure whether explicit frequency loss is a better smoke35 add-on than ControlVAE on the PixArt route.",
            "pipeline": [
                "PixArt transformer backbone",
                "PGD warm start",
                "LAO latent optimization",
                "explicit frequency alignment loss",
                "export transfer / visual / frequency metrics",
            ],
            "candidate_modules": {
                "enabled": ["PixArtAlpha transformer", "vae", "explicit frequency loss"],
                "disabled": ["ControlVAE", "cross-attn regularizer", "self-attn regularizer"],
            },
            "input_dataset": str(SMOKE35_FAKE_ROOT),
            "clean_dataroot": str(CLEAN_DATAROOT),
            "output_dataset": str(OUTPUT_DATAROOT),
            "output_logs": str(RUN_DIR),
            "backbone": "pixart",
            "implementation": "pixart",
            "surrogate": "S",
            "controlvae": 0,
            "lambda_f": ATTACK_ARGS["lambda_f"],
            "cross_attn": "off",
            "self_attn": "off",
            "pseudo_label_source": "smoke35_pseudo_labels",
            "attack_args": ATTACK_ARGS,
        },
    )

    cmd = [str(PIXART_PYTHON), str(PIXART_ROOT / "main.py")]
    for key, value in ATTACK_ARGS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    run_cmd(cmd, cwd=PIXART_ROOT)

    if log_path.exists():
        with log_path.open("rb") as src:
            src.seek(start_size)
            (RUN_DIR / log_path.name).write_bytes(src.read())
    ensure_symlink(RUN_DIR / "attack_dataroot", OUTPUT_DATAROOT)
    for name in ["manifest_real.csv", "manifest_fake.csv", "manifest_fake_adv.csv", "summary_metrics.json"]:
        copy_text_like(OUTPUT_DATAROOT / name, RUN_DIR / name)


def export_metrics() -> None:
    export_csv = RUN_DIR / "adv_logits_surS.csv"
    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "export_detector_logits.py"),
            "--dataroot",
            str(OUTPUT_DATAROOT),
            "--models",
            "E,R,D,S",
            "--batch",
            "32",
            "--out_csv",
            str(export_csv),
        ],
        cwd=DIFFSR_ROOT,
    )
    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "transfer_asr_matrix.py"),
            "--run",
            f"S={export_csv}",
            "--out_csv",
            str(RUN_DIR / "transfer_asr.csv"),
            "--by_subset_csv",
            str(RUN_DIR / "transfer_asr_by_subset.csv"),
        ],
        cwd=DIFFSR_ROOT,
    )
    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "eval_visual_quality.py"),
            "--clean_dataroot",
            str(CLEAN_DATAROOT),
            "--adv_dataroot",
            str(OUTPUT_DATAROOT),
            "--out_dir",
            str(RUN_DIR / "metrics" / "visual"),
            "--save_per_image",
        ],
        cwd=DIFFSR_ROOT,
    )

    freq_root = RUN_DIR / "_frequency_root"
    freq_root.mkdir(parents=True, exist_ok=True)
    ensure_symlink(freq_root / OUTPUT_DATAROOT.name, OUTPUT_DATAROOT)
    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "eval_frequency_metrics.py"),
            "--clean_dataroot",
            str(CLEAN_DATAROOT),
            "--root",
            str(freq_root),
            "--out_dir",
            str(RUN_DIR / "metrics" / "frequency"),
            "--surrogates",
            "S",
            "--res",
            "224",
        ],
        cwd=DIFFSR_ROOT,
    )


def collect_summary() -> dict[str, object]:
    summary_metrics = json.loads((OUTPUT_DATAROOT / "summary_metrics.json").read_text(encoding="utf-8"))
    transfer_row = read_csv_dicts(RUN_DIR / "transfer_asr.csv")[0]
    visual_row = read_csv_dicts(RUN_DIR / "metrics" / "visual" / "visual_quality.csv")[0]
    freq_rows = read_csv_dicts(RUN_DIR / "metrics" / "frequency" / "frequency_metrics.csv")
    freq_row = next(row for row in freq_rows if row["dataset"] == "surS")
    row = {
        "run_id": RUN_ID,
        "backbone": "pixart",
        "implementation": "pixart",
        "surrogate": "S",
        "scope": "smoke_fixed35",
        "num_fake_images": 35,
        "primary_change": "frequency_loss_only",
        "pseudo_label_source": "smoke35_pseudo_labels",
        "pseudo_label_csv": str(PSEUDO_LABEL_CSV),
        "attack_out_dir": str(RUN_DIR),
        "adv_dataroot": str(OUTPUT_DATAROOT),
        "whitebox_asr": float(transfer_row["whitebox_asr"]),
        "blackbox_mean_asr": float(transfer_row["blackbox_mean_asr"]),
        "asr_to_E": float(transfer_row["asr_to_E"]) if transfer_row["asr_to_E"] else "",
        "asr_to_R": float(transfer_row["asr_to_R"]) if transfer_row["asr_to_R"] else "",
        "asr_to_D": float(transfer_row["asr_to_D"]) if transfer_row["asr_to_D"] else "",
        "asr_to_S": float(transfer_row["asr_to_S"]) if transfer_row["asr_to_S"] else "",
        "lpips": float(visual_row["lpips_mean"]),
        "psnr": float(visual_row["psnr_mean"]),
        "ssim": float(visual_row["ssim_mean"]),
        "fourier_l2_to_real": float(freq_row["img_logamp_l2_to_real"]),
        "residual_fourier_l2_to_real": float(freq_row["residual_logamp_l2_to_real"]),
        "attack_mean_psnr": float(summary_metrics["mean_psnr"]),
        "attack_mean_ssim": float(summary_metrics["mean_ssim"]),
        "lambda_f": ATTACK_ARGS["lambda_f"],
        "self_attn": "off",
        "cross_attn": "off",
        "controlvae": "off",
        "compared_to": "pixart_noctrl_700_surS|pixart_ctrl_700_surS|pixart_ctrlvae_pixartproto_smoke35_surS_20260324",
        "evidence": str(RUN_SUMMARY),
    }
    write_json(RUN_SUMMARY, row)
    return row


def main() -> int:
    run_attack()
    export_metrics()
    collect_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
