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
FULL_FAKE_ROOT = CLEAN_DATAROOT / "fake"
ATTACK_DATASETS_ROOT = DATASET_ROOT / "pixart_alpha" / "attack_datasets" / "full_700"
PIXART_PYTHON = REPO_ROOT / "conda_envs" / "pixart_alpha" / "bin" / "python"
PIXART_MODEL_PATH = REPO_ROOT / "checkpoints" / "hf_models" / "pixart-alpha-xl-2-512x512"
PSEUDO_LABEL_CSV = DIFFSR_ROOT / "metadata" / "full700_pseudo_labels.csv"
EXPERIMENT_LOG_CSV = DIFFSR_ROOT / "results" / "experiment_record_log.csv"
DATE_TAG = "20260409"
RESULT_ROOT = PIXART_ROOT / "results" / f"objective_schedule_full700_{DATE_TAG}"
RUNS_ROOT = RESULT_ROOT / "runs"

FIELDNAMES = [
    "record_order",
    "record_id",
    "status",
    "project",
    "family",
    "experiment_name",
    "date_hint",
    "scope",
    "data_n",
    "result_path",
    "backbone",
    "surrogate",
    "controlvae",
    "iterations",
    "t_start",
    "lr",
    "lambda_perc",
    "primary_change",
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
    "compared_to",
    "new_information",
    "evidence",
]

BASE_ATTACK_ARGS = {
    "output_mode": "final",
    "clean_dataroot": str(CLEAN_DATAROOT),
    "real_mode": "symlink",
    "save_origin": 0,
    "mode": "double",
    "images_root": str(FULL_FAKE_ROOT),
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

RUN_SPECS = {
    "two_stage_c2s": {
        "run_id": f"pixart_schedule_full700_surS_two_stage_c2s_{DATE_TAG}",
        "record_id": f"pixart_schedule_full700_surS_two_stage_c2s_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha two-stage cross-to-self objective schedule full700 surrogate S",
        "primary_change": "objective_schedule_two_stage_cross_to_self",
        "schedule_mode": "two_stage_cross_to_self",
        "self_attn_loss_weight": 1.0,
        "cross_attn_loss_weight": 10000.0,
    },
    "two_stage_s2c": {
        "run_id": f"pixart_schedule_full700_surS_two_stage_s2c_{DATE_TAG}",
        "record_id": f"pixart_schedule_full700_surS_two_stage_s2c_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha two-stage self-to-cross objective schedule full700 surrogate S",
        "primary_change": "objective_schedule_two_stage_self_to_cross",
        "schedule_mode": "two_stage_self_to_cross",
        "self_attn_loss_weight": 1.0,
        "cross_attn_loss_weight": 10000.0,
    },
}


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


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


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return FIELDNAMES, []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or FIELDNAMES), rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_or_replace_records(new_rows: list[dict[str, str]]) -> None:
    header, rows = load_csv_rows(EXPERIMENT_LOG_CSV)
    if header != FIELDNAMES:
        header = FIELDNAMES
    by_id = {row["record_id"]: idx for idx, row in enumerate(rows) if row.get("record_id")}
    next_order = max(int(r["record_order"]) for r in rows if r.get("record_order")) if rows else 0
    for row in new_rows:
        idx = by_id.get(row["record_id"])
        if idx is None:
            next_order += 1
            row["record_order"] = str(next_order)
            rows.append(row)
            by_id[row["record_id"]] = len(rows) - 1
        else:
            row["record_order"] = rows[idx]["record_order"]
            rows[idx] = row
    rows.sort(key=lambda r: int(r["record_order"]))
    write_csv_rows(EXPERIMENT_LOG_CSV, FIELDNAMES, rows)


def load_existing_records() -> dict[str, dict[str, str]]:
    _, rows = load_csv_rows(EXPERIMENT_LOG_CSV)
    return {row["record_id"]: row for row in rows if row.get("record_id")}


def run_dir(key: str) -> Path:
    return RUNS_ROOT / RUN_SPECS[key]["run_id"]


def output_dataroot(key: str) -> Path:
    return ATTACK_DATASETS_ROOT / f"PixArtAlpha_full700_adv_schedule_{key}_surS_{DATE_TAG}"


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


def ensure_prerequisites() -> None:
    if not PSEUDO_LABEL_CSV.exists():
        raise FileNotFoundError(f"Missing full700 pseudo-label file: {PSEUDO_LABEL_CSV}")


def run_attack(key: str) -> None:
    ensure_prerequisites()
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
    write_json(
        out_dir / "run_meta.json",
        {
            "run_id": spec["run_id"],
            "record_id": spec["record_id"],
            "problem": "Test whether iteration-wise self/cross attention scheduling improves the fixed-objective tradeoff on PixArt no-ControlVAE full700 surrogate S.",
            "pipeline": [
                "PixArt transformer backbone",
                "DDIM inversion",
                "pseudo-label text forwarding",
                "LAO latent optimization with iteration-wise self/cross objective schedule",
                "export transfer / visual / frequency metrics",
            ],
            "candidate_modules": {
                "enabled": [
                    "PixArtAlpha transformer",
                    "pseudo-label prompt embeddings",
                    "self-attn regularizer",
                    "cross-attn regularizer",
                    "objective scheduler",
                ],
                "disabled": ["ControlVAE", "frequency loss"],
            },
            "metrics": [
                "whitebox_asr",
                "blackbox_mean_asr",
                "asr_to_E",
                "asr_to_R",
                "asr_to_D",
                "lpips",
                "psnr",
                "ssim",
                "fourier_l2_to_real",
                "residual_fourier_l2_to_real",
                "attack_mean_psnr",
                "attack_mean_ssim",
            ],
            "input_dataset": str(FULL_FAKE_ROOT),
            "clean_dataroot": str(CLEAN_DATAROOT),
            "output_dataset": str(adv_dataroot),
            "output_logs": str(out_dir),
            "backbone": "pixart",
            "implementation": "pixart",
            "surrogate": "S",
            "controlvae": 0,
            "schedule_mode": spec["schedule_mode"],
            "schedule_switch_ratio": 0.5,
            "prompt_source": "full700_pseudo_labels",
            "compared_to": "pixart_selfattn_full700_surS_g1p0_20260406|pixart_crossattn_full700_surS_20260409|pixart_schedule_smoke35_surS_two_stage_c2s_20260406|pixart_schedule_smoke35_surS_two_stage_s2c_20260406|pixart_noctrl_700_surS",
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
    for name in ["manifest_real.csv", "manifest_fake.csv", "manifest_fake_adv.csv", "summary_metrics.json", "meta.txt"]:
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
    row = {
        "run_id": spec["run_id"],
        "record_id": spec["record_id"],
        "backbone": "pixart",
        "implementation": "pixart",
        "surrogate": "S",
        "scope": "full_700_fake",
        "num_fake_images": 700,
        "controlvae": 0,
        "schedule_mode": spec["schedule_mode"],
        "schedule_switch_ratio": 0.5,
        "attack_out_dir": str(out_dir),
        "adv_dataroot": str(adv_dataroot),
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
        "compared_to": "pixart_selfattn_full700_surS_g1p0_20260406|pixart_crossattn_full700_surS_20260409|pixart_schedule_smoke35_surS_two_stage_c2s_20260406|pixart_schedule_smoke35_surS_two_stage_s2c_20260406|pixart_noctrl_700_surS",
        "evidence": str(out_dir / "run_summary.json"),
    }
    write_json(out_dir / "run_summary.json", row)
    return row


def build_new_information(summary: dict[str, object], existing_rows: dict[str, dict[str, str]]) -> str:
    self_row = existing_rows["pixart_selfattn_full700_surS_g1p0_20260406"]
    cross_row = existing_rows["pixart_crossattn_full700_surS_20260409"]
    self_blackbox = float(self_row["blackbox_mean_asr"])
    cross_blackbox = float(cross_row["blackbox_mean_asr"])
    self_psnr = float(self_row["psnr"])
    cross_psnr = float(cross_row["psnr"])
    self_fourier = float(self_row["fourier_l2_to_real"])
    cross_fourier = float(cross_row["fourier_l2_to_real"])
    blackbox = float(summary["blackbox_mean_asr"])
    psnr = float(summary["psnr"])
    fourier = float(summary["fourier_l2_to_real"])
    schedule_mode = str(summary["schedule_mode"])

    if blackbox >= self_blackbox and psnr >= cross_psnr:
        headline = "The schedule reaches or exceeds the fixed self-attn blackbox ASR while retaining at least the fixed cross-attn PSNR under the matched full700 setup."
    elif blackbox >= self_blackbox:
        headline = "The schedule preserves self-attn-level attackability but does not recover the full visual advantage of fixed cross-attn."
    elif psnr >= cross_psnr:
        headline = "The schedule preserves cross-attn-level visual quality but does not surpass the fixed self-attn transfer baseline."
    else:
        headline = "The schedule remains an intermediate tradeoff point and does not dominate the fixed full700 baselines."

    return (
        f"{headline} "
        f"Mode={schedule_mode}; blackbox ASR={blackbox:.6f} versus self-attn {self_blackbox:.6f} and cross-attn {cross_blackbox:.6f}; "
        f"PSNR={psnr:.6f} versus self-attn {self_psnr:.6f} and cross-attn {cross_psnr:.6f}; "
        f"Fourier L2={fourier:.6f} versus self-attn {self_fourier:.6f} and cross-attn {cross_fourier:.6f}."
    )


def build_record_row(summary: dict[str, object], existing_rows: dict[str, dict[str, str]]) -> dict[str, str]:
    return {
        "record_order": "",
        "record_id": str(summary["record_id"]),
        "status": "complete",
        "project": "StealthDiffusion_PixArtAlpha",
        "family": "ablation_objective_schedule_full700",
        "experiment_name": RUN_SPECS[str(summary["schedule_mode"]).replace("two_stage_cross_to_self", "two_stage_c2s").replace("two_stage_self_to_cross", "two_stage_s2c")]["experiment_name"],
        "date_hint": DATE_TAG,
        "scope": "full_700_fake",
        "data_n": "700",
        "result_path": str(summary["attack_out_dir"]),
        "backbone": "pixart",
        "surrogate": "S",
        "controlvae": "0",
        "iterations": "5",
        "t_start": "18",
        "lr": "1e-3",
        "lambda_perc": "0.5",
        "primary_change": "objective_schedule_" + str(summary["schedule_mode"]),
        "whitebox_asr": fmt(summary["whitebox_asr"]),
        "blackbox_mean_asr": fmt(summary["blackbox_mean_asr"]),
        "asr_to_E": fmt(summary["asr_to_E"]),
        "asr_to_R": fmt(summary["asr_to_R"]),
        "asr_to_D": fmt(summary["asr_to_D"]),
        "asr_to_S": fmt(summary["asr_to_S"]),
        "lpips": fmt(summary["lpips"]),
        "psnr": fmt(summary["psnr"]),
        "ssim": fmt(summary["ssim"]),
        "fourier_l2_to_real": fmt(summary["fourier_l2_to_real"]),
        "residual_fourier_l2_to_real": fmt(summary["residual_fourier_l2_to_real"]),
        "attack_mean_psnr": fmt(summary["attack_mean_psnr"]),
        "attack_mean_ssim": fmt(summary["attack_mean_ssim"]),
        "compared_to": str(summary["compared_to"]),
        "new_information": build_new_information(summary, existing_rows),
        "evidence": str(summary["evidence"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="two_stage_c2s,two_stage_s2c", help="Comma-separated subset of run keys")
    parser.add_argument("--record-only", action="store_true", help="Only rebuild experiment_record_log.csv rows from existing summaries")
    args = parser.parse_args()

    existing_rows = load_existing_records()
    needed = ["pixart_selfattn_full700_surS_g1p0_20260406", "pixart_crossattn_full700_surS_20260409"]
    if not args.record_only:
        for record_id in needed:
            if record_id not in existing_rows:
                raise KeyError(f"Missing required baseline record before running schedule full700: {record_id}")

    run_keys = [item.strip() for item in args.runs.split(",") if item.strip()]
    summaries: list[dict[str, object]] = []
    for key in run_keys:
        if key not in RUN_SPECS:
            raise ValueError(f"Unsupported run key: {key}")
        if args.record_only:
            summary_path = run_dir(key) / "run_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing summary for run {key}: {summary_path}")
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        run_attack(key)
        export_metrics(key)
        summaries.append(collect_summary(key))

    existing_rows = load_existing_records()
    append_or_replace_records([build_record_row(summary, existing_rows) for summary in summaries])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
