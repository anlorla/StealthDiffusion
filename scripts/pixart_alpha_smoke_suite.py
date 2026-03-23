#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/root/gpufree-data")
PIXART_ROOT = REPO_ROOT / "StealthDiffusion_PixArtAlpha"
DIFFSR_ROOT = REPO_ROOT / "DiffSRAttack"
DATASET_ROOT = REPO_ROOT / "dataset"
SMOKE_FAKE_ROOT = DATASET_ROOT / "StealthDiffusion_SR" / "smoke_fixed7" / "fake"
CLEAN_DATAROOT = DATASET_ROOT / "original_genimage_eval_1400"
ATTACK_DATASETS_ROOT = DATASET_ROOT / "pixart_alpha" / "attack_datasets"
SUITE_ROOT = PIXART_ROOT / "results" / "smoke_suite_fixed7_20260322"
RUNS_ROOT = SUITE_ROOT / "runs"
SUMMARY_CSV = SUITE_ROOT / "suite_summary.csv"
SUMMARY_JSON = SUITE_ROOT / "suite_summary.json"

PIXART_MODEL_PATH = REPO_ROOT / "checkpoints" / "hf_models" / "pixart-alpha-xl-2-512x512"
PIXART_PYTHON = REPO_ROOT / "conda_envs" / "pixart_alpha" / "bin" / "python"

BASE_ATTACK_ARGS = {
    "output_mode": "final",
    "clean_dataroot": str(CLEAN_DATAROOT),
    "real_mode": "symlink",
    "save_origin": 0,
    "mode": "double",
    "images_root": str(SMOKE_FAKE_ROOT),
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
}

RUN_SPECS = {
    "pixart_noctrl_surE": {
        "backbone": "pixart",
        "surrogate": "E",
        "controlvae": 0,
        "model_name": "E,R,D,S",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "sd_noctrl_surE_smoke7|sr_noctrl_surE_smoke7",
    },
    "pixart_noctrl_surR": {
        "backbone": "pixart",
        "surrogate": "R",
        "controlvae": 0,
        "model_name": "R,E,D,S",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "sd_noctrl_surR_smoke7|sr_noctrl_surR_smoke7",
    },
    "pixart_noctrl_surD": {
        "backbone": "pixart",
        "surrogate": "D",
        "controlvae": 0,
        "model_name": "D,E,R,S",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "sd_noctrl_surD_smoke7|sr_noctrl_surD_smoke7",
    },
    "pixart_noctrl_surS": {
        "backbone": "pixart",
        "surrogate": "S",
        "controlvae": 0,
        "model_name": "S,E,R,D",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "sd_noctrl_surS_smoke7|sr_noctrl_surS_smoke7",
    },
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, start=link_path.parent)
    os.symlink(rel, link_path)


def attack_log_path(spec: dict[str, object]) -> Path:
    model_name = str(spec["model_name"]).split(",")
    surrogate = str(spec["surrogate"])
    is_encoder = int(spec["controlvae"])
    name = "".join(model_name) + str(is_encoder)
    return PIXART_ROOT / "exp" / "results" / f"attack_{name}_sur{surrogate}.log"


def run_output_name(run_id: str) -> str:
    return f"StealthDiffusion_smoke7_{run_id}_20260322"


def run_output_dataroot(run_id: str) -> Path:
    return ATTACK_DATASETS_ROOT / run_output_name(run_id)


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def attack_args_for_run(run_id: str) -> dict[str, object]:
    spec = RUN_SPECS[run_id]
    args = dict(BASE_ATTACK_ARGS)
    args["model_name"] = spec["model_name"]
    args["is_encoder"] = int(spec["controlvae"])
    args["pretrained_diffusion_path"] = spec["pretrained_diffusion_path"]
    args["out_dataroot"] = str(run_output_dataroot(run_id))
    args["intermediate_root"] = str(run_dir(run_id) / "intermediates")
    return args


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_attack(run_id: str) -> None:
    spec = RUN_SPECS[run_id]
    out_dir = run_dir(run_id)
    adv_dataroot = run_output_dataroot(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    ATTACK_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    if adv_dataroot.exists():
        raise FileExistsError(f"Output dataroot already exists: {adv_dataroot}")

    args = attack_args_for_run(run_id)
    log_path = attack_log_path(spec)
    start_size = log_path.stat().st_size if log_path.exists() else 0

    run_meta = {
        "run_id": run_id,
        "problem": "Run PixArt-Alpha no-ControlVAE smoke7 to compare against matched SD/SR no-control baselines.",
        "pipeline": [
            "load PixArt-Alpha backbone",
            "encode image with shared VAE",
            "DDIM inversion",
            "latent optimization / LAO",
            "DDIM denoise",
            "decode with VAE",
            "evaluate transfer / visual / frequency metrics",
        ],
        "candidate_modules": {
            "enabled": ["PixArtAlphaPipeline.transformer", "vae", "text_encoder", "detectors E/R/D/S"],
            "disabled": ["ControlVAE NewEncoder", "ControlVAE NewDecoder"],
        },
        "metrics": [
            "whitebox_asr",
            "blackbox_mean_asr",
            "asr_to_E",
            "asr_to_R",
            "asr_to_D",
            "asr_to_S",
            "lpips",
            "psnr",
            "ssim",
            "fourier_l2_to_real",
            "residual_fourier_l2_to_real",
            "attack_mean_psnr",
            "attack_mean_ssim",
        ],
        "input_dataset": str(SMOKE_FAKE_ROOT),
        "clean_dataroot": str(CLEAN_DATAROOT),
        "output_dataset": str(adv_dataroot),
        "output_logs": str(out_dir),
        "output_weights": "none",
        "backbone": spec["backbone"],
        "surrogate": spec["surrogate"],
        "controlvae": spec["controlvae"],
        "compared_to": spec["compared_to"],
        "hyperparams": {
            "t_start": 100,
            "start_step": args["start_step"],
            "diffusion_steps": args["diffusion_steps"],
            "iterations": args["iterations"],
            "lr": args["lr"],
            "guidance": args["guidance"],
            "lambda_lpips": args["lambda_lpips"],
        },
    }
    write_json(out_dir / "run_meta.json", run_meta)

    cmd = [str(PIXART_PYTHON), str(PIXART_ROOT / "main.py")]
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    run_cmd(cmd, cwd=PIXART_ROOT)

    log_copy = out_dir / attack_log_path(spec).name
    if log_path.exists():
        with log_path.open("rb") as src:
            src.seek(start_size)
            log_copy.write_bytes(src.read())

    ensure_symlink(out_dir / "attack_dataroot", adv_dataroot)


def export_metrics(run_id: str) -> None:
    spec = RUN_SPECS[run_id]
    surrogate = str(spec["surrogate"])
    out_dir = run_dir(run_id)
    adv_dataroot = run_output_dataroot(run_id)

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

    visual_dir = out_dir / "metrics" / "visual"
    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "eval_visual_quality.py"),
            "--clean_dataroot",
            str(CLEAN_DATAROOT),
            "--adv_dataroot",
            str(adv_dataroot),
            "--out_dir",
            str(visual_dir),
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


def collect_run_summary(run_id: str) -> dict[str, object]:
    spec = RUN_SPECS[run_id]
    surrogate = str(spec["surrogate"])
    out_dir = run_dir(run_id)
    summary_metrics = json.loads((run_output_dataroot(run_id) / "summary_metrics.json").read_text())
    transfer_row = read_csv_dicts(out_dir / "transfer_asr.csv")[0]
    visual_row = read_csv_dicts(out_dir / "metrics" / "visual" / "visual_quality.csv")[0]
    freq_rows = read_csv_dicts(out_dir / "metrics" / "frequency" / "frequency_metrics.csv")
    freq_row = next(row for row in freq_rows if row["dataset"] == f"sur{surrogate}")
    row = {
        "run_id": run_id,
        "backbone": spec["backbone"],
        "surrogate": surrogate,
        "controlvae": int(spec["controlvae"]),
        "adv_dataroot": str(run_output_dataroot(run_id)),
        "log_copy": str(out_dir / attack_log_path(spec).name),
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
        "compared_to": spec["compared_to"],
    }
    write_json(out_dir / "run_summary.json", row)
    return row


def summarize_completed_runs() -> None:
    rows = []
    for run_id in RUN_SPECS:
        run_summary = run_dir(run_id) / "run_summary.json"
        if run_summary.exists():
            rows.append(json.loads(run_summary.read_text()))

    if not rows:
        return

    fieldnames = [
        "run_id",
        "backbone",
        "surrogate",
        "controlvae",
        "whitebox_asr",
        "blackbox_mean_asr",
        "asr_to_E",
        "asr_to_R",
        "asr_to_D",
        "asr_to_S",
        "lpips",
        "psnr",
        "ssim",
        "fourier_l2_to_real",
        "residual_fourier_l2_to_real",
        "adv_dataroot",
        "log_copy",
        "attack_mean_psnr",
        "attack_mean_ssim",
        "compared_to",
    ]
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    write_json(SUMMARY_JSON, {"rows": rows, "summary_csv": str(SUMMARY_CSV)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUN_SPECS.keys()))
    ap.add_argument("--skip_attack", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    if not args.skip_attack:
        run_attack(args.run)
    if not args.skip_eval:
        export_metrics(args.run)
        collect_run_summary(args.run)
        summarize_completed_runs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
