#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
ATTACK_DATASETS_ROOT = DATASET_ROOT / "pixart_alpha" / "attack_datasets" / "smoke35"
PIXART_PYTHON = REPO_ROOT / "conda_envs" / "pixart_alpha" / "bin" / "python"
PIXART_MODEL_PATH = REPO_ROOT / "checkpoints" / "hf_models" / "pixart-alpha-xl-2-512x512"
PSEUDO_LABEL_CSV = DIFFSR_ROOT / "metadata" / "smoke35_pseudo_labels.csv"
DATE_TAG = "20260328"
RESULT_ROOT = PIXART_ROOT / "results" / f"self_attn_smoke35_matrix_{DATE_TAG}"
RUNS_ROOT = RESULT_ROOT / "runs"

BASE_ATTACK_ARGS = {
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
    "attack_only_fake": 1,
    "seed": 42,
    "save_intermediates": True,
    "pseudo_label_csv": str(PSEUDO_LABEL_CSV),
    "is_encoder": 0,
    "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
    "self_attn_max_layers": 5,
}

RUN_SPECS = {
    "E": {
        "run_id": f"pixart_selfattn_g1p0_surE_{DATE_TAG}",
        "record_id": f"pixart_selfattn_smoke35_surE_g1p0_{DATE_TAG}",
        "model_name": "E,R,D,S",
        "compared_to": "pixart_noctrl_700_surE|pixart_selfattn_smoke35_surS_g1p0_20260326",
    },
    "R": {
        "run_id": f"pixart_selfattn_g1p0_surR_{DATE_TAG}",
        "record_id": f"pixart_selfattn_smoke35_surR_g1p0_{DATE_TAG}",
        "model_name": "R,E,D,S",
        "compared_to": "pixart_noctrl_700_surR|pixart_selfattn_smoke35_surS_g1p0_20260326",
    },
    "D": {
        "run_id": f"pixart_selfattn_g1p0_surD_{DATE_TAG}",
        "record_id": f"pixart_selfattn_smoke35_surD_g1p0_{DATE_TAG}",
        "model_name": "D,E,R,S",
        "compared_to": "pixart_noctrl_700_surD|pixart_selfattn_smoke35_surS_g1p0_20260326",
    },
    "S": {
        "run_id": f"pixart_selfattn_g1p0_surS_{DATE_TAG}",
        "record_id": f"pixart_selfattn_smoke35_surS_g1p0_{DATE_TAG}",
        "model_name": "S,E,R,D",
        "compared_to": "pixart_noctrl_700_surS|pixart_selfattn_smoke35_surS_g1p0_20260326",
    },
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


def run_dir(surrogate: str) -> Path:
    return RUNS_ROOT / RUN_SPECS[surrogate]["run_id"]


def output_dataroot(surrogate: str) -> Path:
    run_id = RUN_SPECS[surrogate]["run_id"]
    return ATTACK_DATASETS_ROOT / f"PixArtAlpha_smoke35_adv_selfattn_g1p0_sur{surrogate}_{DATE_TAG}"


def attack_log_path(surrogate: str) -> Path:
    return PIXART_ROOT / "exp" / "results" / f"attack_{RUN_SPECS[surrogate]['model_name'].replace(',', '')}0_sur{surrogate}.log"


def attack_args(surrogate: str) -> dict[str, object]:
    args = dict(BASE_ATTACK_ARGS)
    args["model_name"] = RUN_SPECS[surrogate]["model_name"]
    args["out_dataroot"] = str(output_dataroot(surrogate))
    args["intermediate_root"] = str(run_dir(surrogate) / "intermediates")
    args["self_attn_loss_weight"] = 1.0
    return args


def run_attack(surrogate: str) -> None:
    out_dir = run_dir(surrogate)
    adv_dataroot = output_dataroot(surrogate)
    out_dir.mkdir(parents=True, exist_ok=True)
    ATTACK_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    if adv_dataroot.exists():
        raise FileExistsError(f"Output dataroot already exists: {adv_dataroot}")

    args = attack_args(surrogate)
    log_path = attack_log_path(surrogate)
    start_size = log_path.stat().st_size if log_path.exists() else 0
    write_json(
        out_dir / "run_meta.json",
        {
            "run_id": RUN_SPECS[surrogate]["run_id"],
            "problem": "Evaluate PixArt self-attn-only smoke35 on a chosen surrogate.",
            "pipeline": [
                "PixArt transformer backbone",
                "DDIM inversion",
                "cache self-attn references",
                "LAO latent optimization with self-attn regularization",
                "export transfer / visual / frequency metrics",
            ],
            "candidate_modules": {
                "enabled": ["PixArtAlpha transformer", "self-attn regularizer"],
                "disabled": ["ControlVAE", "cross-attn regularizer", "frequency loss"],
            },
            "input_dataset": str(SMOKE35_FAKE_ROOT),
            "clean_dataroot": str(CLEAN_DATAROOT),
            "output_dataset": str(adv_dataroot),
            "output_logs": str(out_dir),
            "backbone": "pixart",
            "implementation": "pixart",
            "surrogate": surrogate,
            "controlvae": 0,
            "self_attn_gamma": 1.0,
            "cross_attn": "off",
            "self_attn": "on",
            "pseudo_label_source": "smoke35_pseudo_labels",
            "attack_args": args,
        },
    )

    cmd = [str(PIXART_PYTHON), str(PIXART_ROOT / "main.py")]
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    run_cmd(cmd, cwd=PIXART_ROOT)

    if log_path.exists():
        with log_path.open("rb") as src:
            src.seek(start_size)
            (out_dir / log_path.name).write_bytes(src.read())
    ensure_symlink(out_dir / "attack_dataroot", adv_dataroot)
    for name in ["manifest_real.csv", "manifest_fake.csv", "manifest_fake_adv.csv", "summary_metrics.json"]:
        copy_text_like(adv_dataroot / name, out_dir / name)


def export_metrics(surrogate: str) -> None:
    out_dir = run_dir(surrogate)
    adv_dataroot = output_dataroot(surrogate)
    export_csv = out_dir / f"adv_logits_sur{surrogate}.csv"
    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "export_detector_logits.py"),
            "--dataroot",
            str(adv_dataroot),
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
            f"{surrogate}={export_csv}",
            "--out_csv",
            str(out_dir / "transfer_asr.csv"),
            "--by_subset_csv",
            str(out_dir / "transfer_asr_by_subset.csv"),
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
            str(adv_dataroot),
            "--out_dir",
            str(out_dir / "metrics" / "visual"),
            "--save_per_image",
        ],
        cwd=DIFFSR_ROOT,
    )
    freq_root = out_dir / "_frequency_root"
    freq_root.mkdir(parents=True, exist_ok=True)
    ensure_symlink(freq_root / adv_dataroot.name, adv_dataroot)
    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "eval_frequency_metrics.py"),
            "--clean_dataroot",
            str(CLEAN_DATAROOT),
            "--root",
            str(freq_root),
            "--out_dir",
            str(out_dir / "metrics" / "frequency"),
            "--surrogates",
            surrogate,
            "--res",
            "224",
        ],
        cwd=DIFFSR_ROOT,
    )


def collect_summary(surrogate: str) -> dict[str, object]:
    out_dir = run_dir(surrogate)
    adv_dataroot = output_dataroot(surrogate)
    summary_metrics = json.loads((adv_dataroot / "summary_metrics.json").read_text(encoding="utf-8"))
    transfer_row = read_csv_dicts(out_dir / "transfer_asr.csv")[0]
    visual_row = read_csv_dicts(out_dir / "metrics" / "visual" / "visual_quality.csv")[0]
    freq_rows = read_csv_dicts(out_dir / "metrics" / "frequency" / "frequency_metrics.csv")
    freq_row = next(row for row in freq_rows if row["dataset"] == f"sur{surrogate}")
    row = {
        "run_id": RUN_SPECS[surrogate]["run_id"],
        "backbone": "pixart",
        "impl": "pixart",
        "surrogate": surrogate,
        "controlvae": 0,
        "adv_dataroot": str(adv_dataroot),
        "self_attn_gamma": 1.0,
        "self_attn_layers": [
            "transformer_blocks.0.attn1",
            "transformer_blocks.7.attn1",
            "transformer_blocks.14.attn1",
            "transformer_blocks.21.attn1",
            "transformer_blocks.27.attn1",
        ],
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
    }
    write_json(out_dir / "run_summary.json", row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogates", default="E,R,D", help="Comma-separated subset of E,R,D,S")
    args = parser.parse_args()
    surrogates = [s.strip() for s in args.surrogates.split(",") if s.strip()]
    for surrogate in surrogates:
        if surrogate not in RUN_SPECS:
            raise ValueError(f"Unsupported surrogate: {surrogate}")
        run_attack(surrogate)
        export_metrics(surrogate)
        collect_summary(surrogate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
