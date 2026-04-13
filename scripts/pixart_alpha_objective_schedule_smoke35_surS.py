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
EXPERIMENT_LOG_CSV = DIFFSR_ROOT / "results" / "experiment_record_log.csv"
DATE_TAG = "20260406"
RESULT_ROOT = PIXART_ROOT / "results" / f"objective_schedule_smoke35_{DATE_TAG}"
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
    "cross_attn_max_layers": 5,
    "model_name": "S,E,R,D",
}

HISTORICAL_BASELINE_RECORD_IDS = {
    "self": "pixart_selfattn_smoke35_surS_g1p0_20260326",
    "cross": "pixart_crossattn_smoke35_surS_20260326",
    "noctrl": "pixart_noctrl_700_surS",
}

SCHEDULE_BASELINE_RECORD_IDS = {
    "self": f"pixart_schedule_smoke35_surS_selfonly_{DATE_TAG}",
    "cross": f"pixart_schedule_smoke35_surS_crossonly_{DATE_TAG}",
    "noctrl": "pixart_noctrl_700_surS",
}

RUN_SPECS = {
    "selfonly": {
        "run_id": f"pixart_schedule_selfonly_surS_{DATE_TAG}",
        "record_id": f"pixart_schedule_smoke35_surS_selfonly_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha objective-schedule self-only smoke35 surrogate S",
        "primary_change": "objective_schedule_self_only",
        "schedule_mode": "fixed_self_only",
        "self_attn_loss_weight": 1.0,
        "cross_attn_loss_weight": 0.0,
    },
    "crossonly": {
        "run_id": f"pixart_schedule_crossonly_surS_{DATE_TAG}",
        "record_id": f"pixart_schedule_smoke35_surS_crossonly_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha objective-schedule cross-only smoke35 surrogate S",
        "primary_change": "objective_schedule_cross_only",
        "schedule_mode": "fixed_cross_only",
        "self_attn_loss_weight": 0.0,
        "cross_attn_loss_weight": 10000.0,
    },
    "two_stage_s2c": {
        "run_id": f"pixart_schedule_two_stage_s2c_surS_{DATE_TAG}",
        "record_id": f"pixart_schedule_smoke35_surS_two_stage_s2c_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha two-stage self-to-cross objective schedule smoke35 surrogate S",
        "primary_change": "objective_schedule_two_stage_self_to_cross",
        "schedule_mode": "two_stage_self_to_cross",
        "self_attn_loss_weight": 1.0,
        "cross_attn_loss_weight": 10000.0,
    },
    "two_stage_c2s": {
        "run_id": f"pixart_schedule_two_stage_c2s_surS_{DATE_TAG}",
        "record_id": f"pixart_schedule_smoke35_surS_two_stage_c2s_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha two-stage cross-to-self objective schedule smoke35 surrogate S",
        "primary_change": "objective_schedule_two_stage_cross_to_self",
        "schedule_mode": "two_stage_cross_to_self",
        "self_attn_loss_weight": 1.0,
        "cross_attn_loss_weight": 10000.0,
    },
    "linear_s2c": {
        "run_id": f"pixart_schedule_linear_s2c_surS_{DATE_TAG}",
        "record_id": f"pixart_schedule_smoke35_surS_linear_s2c_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha linear self-to-cross objective schedule smoke35 surrogate S",
        "primary_change": "objective_schedule_linear_self_to_cross",
        "schedule_mode": "linear_self_to_cross",
        "self_attn_loss_weight": 1.0,
        "cross_attn_loss_weight": 10000.0,
    },
    "linear_c2s": {
        "run_id": f"pixart_schedule_linear_c2s_surS_{DATE_TAG}",
        "record_id": f"pixart_schedule_smoke35_surS_linear_c2s_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha linear cross-to-self objective schedule smoke35 surrogate S",
        "primary_change": "objective_schedule_linear_cross_to_self",
        "schedule_mode": "linear_cross_to_self",
        "self_attn_loss_weight": 1.0,
        "cross_attn_loss_weight": 10000.0,
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


def run_dir(key: str) -> Path:
    return RUNS_ROOT / RUN_SPECS[key]["run_id"]


def output_dataroot(key: str) -> Path:
    return ATTACK_DATASETS_ROOT / f"PixArtAlpha_smoke35_adv_schedule_{key}_surS_{DATE_TAG}"


def attack_log_path() -> Path:
    return PIXART_ROOT / "exp" / "results" / "attack_SERD0_surS.log"


def attack_args(key: str) -> dict[str, object]:
    spec = RUN_SPECS[key]
    args = dict(BASE_ATTACK_ARGS)
    args["out_dataroot"] = str(output_dataroot(key))
    args["intermediate_root"] = str(run_dir(key) / "intermediates")
    args["attn_schedule_mode"] = spec["schedule_mode"]
    args["attn_schedule_switch_ratio"] = 0.5
    args["self_attn_loss_weight"] = spec["self_attn_loss_weight"]
    args["cross_attn_loss_weight"] = spec["cross_attn_loss_weight"]
    return args


def load_existing_records() -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, str]]]:
    with EXPERIMENT_LOG_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    by_id = {row["record_id"]: row for row in rows}
    return fieldnames, rows, by_id


def run_attack(key: str) -> None:
    spec = RUN_SPECS[key]
    out_dir = run_dir(key)
    adv_dataroot = output_dataroot(key)
    out_dir.mkdir(parents=True, exist_ok=True)
    ATTACK_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    if adv_dataroot.exists():
        raise FileExistsError(f"Output dataroot already exists: {adv_dataroot}")

    args = attack_args(key)
    log_path = attack_log_path()
    start_size = log_path.stat().st_size if log_path.exists() else 0
    enabled_modules = ["PixArtAlpha transformer", "detector surrogate S"]
    if args["self_attn_loss_weight"] > 0:
        enabled_modules.append("self-attn regularizer")
    if args["cross_attn_loss_weight"] > 0:
        enabled_modules.append("cross-attn regularizer")

    write_json(
        out_dir / "run_meta.json",
        {
            "run_id": spec["run_id"],
            "record_id": spec["record_id"],
            "problem": "Test whether iteration-wise self/cross attention scheduling improves the fixed-objective tradeoff on PixArt no-ControlVAE smoke35 surrogate S.",
            "pipeline": [
                "PixArt transformer backbone",
                "DDIM inversion startpoint",
                "latent optimization with attack/l1/lpips losses",
                f"attention schedule mode={spec['schedule_mode']}",
                "export transfer / visual / frequency metrics",
            ],
            "candidate_modules": {
                "enabled": enabled_modules,
                "disabled": ["ControlVAE", "frequency loss"],
            },
            "input_dataset": str(SMOKE35_FAKE_ROOT),
            "clean_dataroot": str(CLEAN_DATAROOT),
            "output_dataset": str(adv_dataroot),
            "output_logs": str(out_dir),
            "backbone": "pixart",
            "implementation": "pixart",
            "surrogate": "S",
            "controlvae": 0,
            "schedule_mode": spec["schedule_mode"],
            "schedule_switch_ratio": 0.5,
            "self_attn_base_weight": spec["self_attn_loss_weight"],
            "cross_attn_base_weight": spec["cross_attn_loss_weight"],
            "pseudo_label_source": "smoke35_pseudo_labels",
            "attack_args": args,
        },
    )

    cmd = [str(PIXART_PYTHON), str(PIXART_ROOT / "main.py")]
    for arg_key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg_key}")
        else:
            cmd.extend([f"--{arg_key}", str(value)])
    run_cmd(cmd, cwd=PIXART_ROOT)

    if log_path.exists():
        with log_path.open("rb") as src:
            src.seek(start_size)
            (out_dir / log_path.name).write_bytes(src.read())
    ensure_symlink(out_dir / "attack_dataroot", adv_dataroot)
    for name in ["manifest_real.csv", "manifest_fake.csv", "manifest_fake_adv.csv", "summary_metrics.json"]:
        copy_text_like(adv_dataroot / name, out_dir / name)


def export_metrics(key: str) -> None:
    out_dir = run_dir(key)
    adv_dataroot = output_dataroot(key)
    export_csv = out_dir / "adv_logits_surS.csv"
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
            f"S={export_csv}",
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
            "S",
            "--res",
            "224",
        ],
        cwd=DIFFSR_ROOT,
    )


def collect_summary(key: str) -> dict[str, object]:
    spec = RUN_SPECS[key]
    out_dir = run_dir(key)
    adv_dataroot = output_dataroot(key)
    summary_metrics = json.loads((adv_dataroot / "summary_metrics.json").read_text(encoding="utf-8"))
    transfer_row = read_csv_dicts(out_dir / "transfer_asr.csv")[0]
    visual_row = read_csv_dicts(out_dir / "metrics" / "visual" / "visual_quality.csv")[0]
    freq_rows = read_csv_dicts(out_dir / "metrics" / "frequency" / "frequency_metrics.csv")
    freq_row = next(row for row in freq_rows if row["dataset"] == "surS")
    compared_to_ids = (
        HISTORICAL_BASELINE_RECORD_IDS if key in {"selfonly", "crossonly"} else SCHEDULE_BASELINE_RECORD_IDS
    )
    row = {
        "run_id": spec["run_id"],
        "record_id": spec["record_id"],
        "backbone": "pixart",
        "impl": "pixart",
        "surrogate": "S",
        "controlvae": 0,
        "adv_dataroot": str(adv_dataroot),
        "schedule_mode": spec["schedule_mode"],
        "schedule_switch_ratio": 0.5,
        "self_attn_base_weight": spec["self_attn_loss_weight"],
        "cross_attn_base_weight": spec["cross_attn_loss_weight"],
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
        "compared_to": "|".join(
            [
                compared_to_ids["self"],
                compared_to_ids["cross"],
                compared_to_ids["noctrl"],
            ]
        ),
    }
    write_json(out_dir / "run_summary.json", row)
    return row


def build_new_information(key: str, summary: dict[str, object], existing_rows: dict[str, dict[str, str]]) -> str:
    compared_to_ids = (
        HISTORICAL_BASELINE_RECORD_IDS if key in {"selfonly", "crossonly"} else SCHEDULE_BASELINE_RECORD_IDS
    )
    self_row = existing_rows[compared_to_ids["self"]]
    cross_row = existing_rows[compared_to_ids["cross"]]
    self_blackbox = float(self_row["blackbox_mean_asr"])
    cross_blackbox = float(cross_row["blackbox_mean_asr"])
    self_psnr = float(self_row["psnr"])
    cross_psnr = float(cross_row["psnr"])
    self_fourier = float(self_row["fourier_l2_to_real"])
    cross_fourier = float(cross_row["fourier_l2_to_real"])
    blackbox = float(summary["blackbox_mean_asr"])
    psnr = float(summary["psnr"])
    fourier = float(summary["fourier_l2_to_real"])

    if key == "selfonly":
        return (
            "Re-runs the fixed self-only objective under the new schedule-enabled codepath. "
            f"Compared with {compared_to_ids['self']}, blackbox ASR shifts from {self_blackbox:.6f} to {blackbox:.6f}, "
            f"PSNR from {self_psnr:.6f} to {psnr:.6f}, and Fourier L2 from {self_fourier:.6f} to {fourier:.6f}. "
            "This checks that the schedule refactor preserves the original self-attn baseline behavior."
        )
    if key == "crossonly":
        return (
            "Runs a matched no-ControlVAE cross-only baseline inside the new schedule framework. "
            f"Relative to the older cross-attn row {compared_to_ids['cross']}, blackbox ASR shifts from {cross_blackbox:.6f} to {blackbox:.6f}, "
            f"PSNR from {cross_psnr:.6f} to {psnr:.6f}, and Fourier L2 from {cross_fourier:.6f} to {fourier:.6f}. "
            "This indicates the historical cross-attn surrogate-S row is not directly reproduced under the current no-ControlVAE path, so the remaining schedule rows should be judged against the freshly rerun 20260406 self-only and cross-only baselines instead."
        )

    if blackbox >= self_blackbox and psnr >= cross_psnr:
        headline = "The schedule reaches or exceeds the fixed self-only blackbox ASR while retaining at least the fixed cross-only PSNR."
    elif blackbox > cross_blackbox and psnr > self_psnr:
        headline = "The schedule moves into the interior of the self-vs-cross tradeoff, improving attackability over fixed cross-only while improving visual quality over fixed self-only."
    elif blackbox >= self_blackbox:
        headline = "The schedule behaves closer to the self-attn side, retaining strong transfer ASR without recovering the full cross-attn visual gain."
    elif psnr >= cross_psnr:
        headline = "The schedule behaves closer to the cross-attn side, preserving most of the visual gain while not recovering the self-attn transfer strength."
    else:
        headline = "The schedule does not outperform either fixed baseline on the main attack-quality tradeoff and currently looks less promising than the simpler fixed objectives."

    return (
        f"{headline} "
        f"Observed blackbox ASR={blackbox:.6f} versus self-only {self_blackbox:.6f} and cross-only {cross_blackbox:.6f}; "
        f"PSNR={psnr:.6f} versus self-only {self_psnr:.6f} and cross-only {cross_psnr:.6f}; "
        f"Fourier L2={fourier:.6f} versus self-only {self_fourier:.6f} and cross-only {cross_fourier:.6f}."
    )


def append_experiment_record(key: str, summary: dict[str, object]) -> None:
    fieldnames, rows, existing_rows = load_existing_records()
    record_id = RUN_SPECS[key]["record_id"]
    if record_id in existing_rows:
        raise ValueError(f"record_id already exists in experiment log: {record_id}")

    max_order = max(int(row["record_order"]) for row in rows)
    row = {name: "" for name in fieldnames}
    row.update(
        {
            "record_order": str(max_order + 1),
            "record_id": record_id,
            "status": "complete",
            "project": "StealthDiffusion_PixArtAlpha",
            "family": "ablation_objective_schedule",
            "experiment_name": RUN_SPECS[key]["experiment_name"],
            "date_hint": DATE_TAG,
            "scope": "smoke_fixed35",
            "data_n": "35",
            "result_path": str(run_dir(key)),
            "backbone": "pixart",
            "surrogate": "S",
            "controlvae": "0",
            "iterations": str(BASE_ATTACK_ARGS["iterations"]),
            "t_start": str(BASE_ATTACK_ARGS["start_step"]),
            "lr": str(BASE_ATTACK_ARGS["lr"]),
            "lambda_perc": str(BASE_ATTACK_ARGS["lambda_lpips"]),
            "primary_change": RUN_SPECS[key]["primary_change"],
            "whitebox_asr": f"{float(summary['whitebox_asr']):.6f}",
            "blackbox_mean_asr": f"{float(summary['blackbox_mean_asr']):.6f}",
            "asr_to_E": f"{float(summary['asr_to_E']):.6f}" if summary["asr_to_E"] != "" else "",
            "asr_to_R": f"{float(summary['asr_to_R']):.6f}" if summary["asr_to_R"] != "" else "",
            "asr_to_D": f"{float(summary['asr_to_D']):.6f}" if summary["asr_to_D"] != "" else "",
            "asr_to_S": f"{float(summary['asr_to_S']):.6f}" if summary["asr_to_S"] != "" else "",
            "lpips": f"{float(summary['lpips']):.6f}",
            "psnr": f"{float(summary['psnr']):.6f}",
            "ssim": f"{float(summary['ssim']):.6f}",
            "fourier_l2_to_real": f"{float(summary['fourier_l2_to_real']):.6f}",
            "residual_fourier_l2_to_real": f"{float(summary['residual_fourier_l2_to_real']):.6f}",
            "attack_mean_psnr": f"{float(summary['attack_mean_psnr']):.6f}",
            "attack_mean_ssim": f"{float(summary['attack_mean_ssim']):.6f}",
            "compared_to": str(summary["compared_to"]),
            "new_information": build_new_information(key, summary, existing_rows),
            "evidence": str(run_dir(key) / "run_summary.json"),
        }
    )

    with EXPERIMENT_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        default="selfonly,crossonly,two_stage_s2c,two_stage_c2s,linear_s2c,linear_c2s",
        help="Comma-separated subset of run keys.",
    )
    args = parser.parse_args()
    run_keys = [item.strip() for item in args.runs.split(",") if item.strip()]
    for key in run_keys:
        if key not in RUN_SPECS:
            raise ValueError(f"Unsupported run key: {key}")
        run_attack(key)
        export_metrics(key)
        summary = collect_summary(key)
        append_experiment_record(key, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
