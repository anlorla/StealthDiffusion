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
RESULTS_ROOT = DIFFSR_ROOT / "results"
EXPERIMENT_RECORD_CSV = RESULTS_ROOT / "experiment_record_log.csv"
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

DATE_TAG = "20260327"
PIXART_PYTHON = REPO_ROOT / "conda_envs" / "pixart_alpha" / "bin" / "python"
PIXART_MODEL_PATH = REPO_ROOT / "checkpoints" / "hf_models" / "pixart-alpha-xl-2-512x512"
CLEAN_DATAROOT = DATASET_ROOT / "original_genimage_eval_1400"
SMOKE35_FAKE_ROOT = DATASET_ROOT / "smoke_fixed35" / "fake"
ATTACK_DATASETS_ROOT = DATASET_ROOT / "pixart_alpha" / "attack_datasets" / "smoke35"
PSEUDO_LABEL_CSV = DIFFSR_ROOT / "metadata" / "smoke35_pseudo_labels.csv"
RESULT_ROOT = PIXART_ROOT / "results" / f"startpoint_smoke35_{DATE_TAG}"
RUNS_ROOT = RESULT_ROOT / "runs"
SUITE_SUMMARY_JSON = RESULT_ROOT / "suite_summary.json"
SUITE_SUMMARY_CSV = RESULT_ROOT / "suite_summary.csv"
SUITE_ANALYSIS_MD = RESULT_ROOT / "suite_analysis.md"

BASE_ATTACK_ARGS = {
    "output_mode": "final",
    "clean_dataroot": str(CLEAN_DATAROOT),
    "real_mode": "symlink",
    "save_origin": 0,
    "mode": "double",
    "images_root": str(SMOKE35_FAKE_ROOT),
    "diffusion_steps": 20,
    "start_step": 18,
    "startpoint_mode": "ddim_inversion",
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
    "model_name": "S,E,R,D",
    "is_encoder": 0,
    "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
    "self_attn_loss_weight": 0.1,
}

RUN_SPECS = [
    {
        "run_id": "pixart_startpoint_clean_latent_surS",
        "record_id": f"pixart_startpoint_clean_latent_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha clean latent startpoint smoke35 surrogate S",
        "primary_change": "startpoint_clean_latent",
        "startpoint_family": "clean",
        "startpoint_value": "z0",
        "startpoint_mode": "clean_latent",
        "start_step": 20,
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_euler_sigma1_surS",
        "record_id": f"pixart_startpoint_euler_sigma1_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha Euler sigma1 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_euler_sigma_1",
        "startpoint_family": "euler",
        "startpoint_value": "sigma_1",
        "startpoint_mode": "euler_add_noise",
        "start_step": 18,
        "aligned_ddim": "t_start_18",
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_euler_sigma2_surS",
        "record_id": f"pixart_startpoint_euler_sigma2_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha Euler sigma2 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_euler_sigma_2",
        "startpoint_family": "euler",
        "startpoint_value": "sigma_2",
        "startpoint_mode": "euler_add_noise",
        "start_step": 15,
        "aligned_ddim": "t_start_15",
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_euler_sigma3_surS",
        "record_id": f"pixart_startpoint_euler_sigma3_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha Euler sigma3 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_euler_sigma_3",
        "startpoint_family": "euler",
        "startpoint_value": "sigma_3",
        "startpoint_mode": "euler_add_noise",
        "start_step": 12,
        "aligned_ddim": "t_start_12",
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_euler_sigma4_surS",
        "record_id": f"pixart_startpoint_euler_sigma4_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha Euler sigma4 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_euler_sigma_4",
        "startpoint_family": "euler",
        "startpoint_value": "sigma_4",
        "startpoint_mode": "euler_add_noise",
        "start_step": 8,
        "aligned_ddim": "t_start_8",
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_ddim_t8_surS",
        "record_id": f"pixart_startpoint_ddim_t8_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha DDIM inversion t8 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_ddim_t8",
        "startpoint_family": "ddim",
        "startpoint_value": "t_start_8",
        "startpoint_mode": "ddim_inversion",
        "start_step": 8,
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_ddim_t12_surS",
        "record_id": f"pixart_startpoint_ddim_t12_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha DDIM inversion t12 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_ddim_t12",
        "startpoint_family": "ddim",
        "startpoint_value": "t_start_12",
        "startpoint_mode": "ddim_inversion",
        "start_step": 12,
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_ddim_t15_surS",
        "record_id": f"pixart_startpoint_ddim_t15_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha DDIM inversion t15 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_ddim_t15",
        "startpoint_family": "ddim",
        "startpoint_value": "t_start_15",
        "startpoint_mode": "ddim_inversion",
        "start_step": 15,
        "compared_to": "",
    },
    {
        "run_id": "pixart_startpoint_ddim_t18_surS",
        "record_id": f"pixart_startpoint_ddim_t18_smoke35_surS_{DATE_TAG}",
        "experiment_name": "PixArt-Alpha DDIM inversion t18 startpoint smoke35 surrogate S",
        "primary_change": "startpoint_ddim_t18",
        "startpoint_family": "ddim",
        "startpoint_value": "t_start_18",
        "startpoint_mode": "ddim_inversion",
        "start_step": 18,
        "compared_to": "",
    },
]


def fmt(x) -> str:
    if x == "" or x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def run_dir(spec: dict[str, object]) -> Path:
    return RUNS_ROOT / str(spec["run_id"])


def output_dataroot(spec: dict[str, object]) -> Path:
    return ATTACK_DATASETS_ROOT / f"{spec['run_id']}_{DATE_TAG}"


def attack_log_path() -> Path:
    return PIXART_ROOT / "exp" / "results" / "attack_SERD0_surS.log"


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


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


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        raw = list(csv.reader(f))
    header = [x.strip() for x in raw[0]]
    rows = []
    for row in raw[1:]:
        if not row:
            continue
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in zip(header, row)})
    return header, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attack_args(spec: dict[str, object]) -> dict[str, object]:
    args = dict(BASE_ATTACK_ARGS)
    args["start_step"] = int(spec["start_step"])
    args["startpoint_mode"] = str(spec["startpoint_mode"])
    args["out_dataroot"] = str(output_dataroot(spec))
    args["intermediate_root"] = str(run_dir(spec) / "intermediates")
    return args


def export_metrics(spec: dict[str, object]) -> None:
    out_dir = run_dir(spec)
    adv_dataroot = output_dataroot(spec)
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


def collect_summary(spec: dict[str, object]) -> dict[str, object]:
    out_dir = run_dir(spec)
    adv_dataroot = output_dataroot(spec)
    summary_metrics = json.loads((adv_dataroot / "summary_metrics.json").read_text(encoding="utf-8"))
    transfer_row = read_csv_dicts(out_dir / "transfer_asr.csv")[0]
    visual_row = read_csv_dicts(out_dir / "metrics" / "visual" / "visual_quality.csv")[0]
    freq_rows = read_csv_dicts(out_dir / "metrics" / "frequency" / "frequency_metrics.csv")
    freq_row = next(row for row in freq_rows if row["dataset"] == "surS")
    startpoint_meta = json.loads((out_dir / "intermediates" / "startpoint_meta.json").read_text(encoding="utf-8"))
    row = {
        "run_id": str(spec["run_id"]),
        "record_id": str(spec["record_id"]),
        "backbone": "pixart",
        "implementation": "pixart",
        "surrogate": "S",
        "scope": "smoke_fixed35",
        "num_fake_images": 35,
        "primary_change": str(spec["primary_change"]),
        "startpoint_family": str(spec["startpoint_family"]),
        "startpoint_value": str(spec["startpoint_value"]),
        "requested_start_step": int(spec["start_step"]),
        "startpoint_mode": str(spec["startpoint_mode"]),
        "chosen_index": startpoint_meta.get("chosen_index", ""),
        "chosen_timestep": startpoint_meta.get("chosen_timestep", ""),
        "sigma_value": startpoint_meta.get("sigma_value", ""),
        "self_attn": "on",
        "gamma": 0.1,
        "cross_attn": "off",
        "controlvae": "off",
        "pseudo_label_source": "smoke35_pseudo_labels",
        "pseudo_label_csv": str(PSEUDO_LABEL_CSV),
        "attack_out_dir": str(out_dir),
        "adv_dataroot": str(adv_dataroot),
        "s_asr": float(transfer_row["whitebox_asr"]),
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
        "evidence": str(out_dir / "run_summary.json"),
    }
    write_json(out_dir / "run_summary.json", row)
    return row


def compute_suite_scores(rows: list[dict[str, object]]) -> None:
    for metric in ["psnr", "ssim"]:
        vals = [float(r[metric]) for r in rows]
        mn, mx = min(vals), max(vals)
        for r in rows:
            r[f"{metric}_hat"] = 1.0 if mx == mn else (float(r[metric]) - mn) / (mx - mn)
    for metric in ["lpips", "fourier_l2_to_real", "residual_fourier_l2_to_real"]:
        vals = [float(r[metric]) for r in rows]
        mn, mx = min(vals), max(vals)
        for r in rows:
            r[f"{metric}_hat"] = 1.0 if mx == mn else (mx - float(r[metric])) / (mx - mn)
    for r in rows:
        r["v_score"] = (
            float(r["psnr_hat"]) + float(r["ssim_hat"]) + float(r["lpips_hat"])
        ) / 3.0
        r["s_score"] = (
            float(r["fourier_l2_to_real_hat"]) + float(r["residual_fourier_l2_to_real_hat"])
        ) / 2.0


def analyze_suite(rows: list[dict[str, object]]) -> str:
    clean = next(r for r in rows if r["startpoint_family"] == "clean")
    eulers = sorted(
        [r for r in rows if r["startpoint_family"] == "euler"],
        key=lambda r: int(str(r["primary_change"]).rsplit("_", 1)[-1]),
    )
    ddims = sorted(
        [r for r in rows if r["startpoint_family"] == "ddim"],
        key=lambda r: int(str(r["startpoint_value"]).split("_")[-1]),
    )
    best_blackbox = max(rows, key=lambda r: float(r["blackbox_mean_asr"]))
    best_visual = max(rows, key=lambda r: float(r["v_score"]))
    best_spectrum = max(rows, key=lambda r: float(r["s_score"]))

    lines = [
        "# PixArt Startpoint Smoke35 Trends",
        "",
        f"- Best blackbox ASR: `{best_blackbox['run_id']}` = {float(best_blackbox['blackbox_mean_asr']):.6f}",
        f"- Best V-Score: `{best_visual['run_id']}` = {float(best_visual['v_score']):.6f}",
        f"- Best S-Score: `{best_spectrum['run_id']}` = {float(best_spectrum['s_score']):.6f}",
        "",
        "## Trend Notes",
        "",
        f"- Clean latent anchor `{clean['run_id']}` gives blackbox={float(clean['blackbox_mean_asr']):.6f}, PSNR={float(clean['psnr']):.6f}, Fourier={float(clean['fourier_l2_to_real']):.6f}.",
    ]

    if eulers:
        lines.append(
            f"- Euler family blackbox spans {float(min(eulers, key=lambda r: float(r['blackbox_mean_asr']))['blackbox_mean_asr']):.6f} -> {float(max(eulers, key=lambda r: float(r['blackbox_mean_asr']))['blackbox_mean_asr']):.6f}; "
            f"Fourier spans {float(min(eulers, key=lambda r: float(r['fourier_l2_to_real']))['fourier_l2_to_real']):.6f} -> {float(max(eulers, key=lambda r: float(r['fourier_l2_to_real']))['fourier_l2_to_real']):.6f}."
        )
    if ddims:
        lines.append(
            f"- DDIM family blackbox spans {float(min(ddims, key=lambda r: float(r['blackbox_mean_asr']))['blackbox_mean_asr']):.6f} -> {float(max(ddims, key=lambda r: float(r['blackbox_mean_asr']))['blackbox_mean_asr']):.6f}; "
            f"Fourier spans {float(min(ddims, key=lambda r: float(r['fourier_l2_to_real']))['fourier_l2_to_real']):.6f} -> {float(max(ddims, key=lambda r: float(r['fourier_l2_to_real']))['fourier_l2_to_real']):.6f}."
        )
    lines.append(
        "- Compare Euler and DDIM by aligned depths via `sigma1<->t18`, `sigma2<->t15`, `sigma3<->t12`, `sigma4<->t8`."
    )
    return "\n".join(lines) + "\n"


def build_record_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    clean_id = next(r["record_id"] for r in rows if r["startpoint_family"] == "clean")
    out = []
    for r in rows:
        compared_to = clean_id
        if r["record_id"] == clean_id:
            compared_to = "pixart_selfattn_smoke35_surS_g0p1_20260326|pixart_frequency_smoke35_surS_20260327"
        elif r["startpoint_family"] == "euler":
            compared_to = clean_id
            if "aligned_ddim" in next(spec for spec in RUN_SPECS if spec["record_id"] == r["record_id"]):
                aligned = next(spec for spec in RUN_SPECS if spec["record_id"] == r["record_id"])["aligned_ddim"]
                compared_to = f"{clean_id}|pixart_startpoint_ddim_t{aligned.split('_')[-1]}_smoke35_surS_{DATE_TAG}"
        elif r["startpoint_family"] == "ddim":
            compared_to = clean_id

        sigma_part = ""
        if r.get("sigma_value") != "":
            sigma_part = f", sigma={float(r['sigma_value']):.6f}"
        new_information = (
            f"Smoke35 PixArt startpoint ablation with startpoint_family={r['startpoint_family']}, "
            f"startpoint_value={r['startpoint_value']}, startpoint_mode={r['startpoint_mode']}, "
            f"self_attn=on, gamma=0.1, cross_attn=off, controlvae=off, requested_start_step={r['requested_start_step']}, "
            f"chosen_timestep={r['chosen_timestep']}{sigma_part}. "
            f"Observed whitebox/blackbox ASR={float(r['whitebox_asr']):.6f}/{float(r['blackbox_mean_asr']):.6f}, "
            f"PSNR/SSIM/LPIPS={float(r['psnr']):.6f}/{float(r['ssim']):.6f}/{float(r['lpips']):.6f}, "
            f"Fourier/residual Fourier={float(r['fourier_l2_to_real']):.6f}/{float(r['residual_fourier_l2_to_real']):.6f}."
        )
        row = {k: "" for k in FIELDNAMES}
        row.update(
            {
                "record_id": str(r["record_id"]),
                "status": "complete",
                "project": "StealthDiffusion_PixArtAlpha",
                "family": "ablation_startpoint",
                "experiment_name": str(next(spec["experiment_name"] for spec in RUN_SPECS if spec["record_id"] == r["record_id"])),
                "date_hint": DATE_TAG,
                "scope": "smoke_fixed35",
                "data_n": "35",
                "result_path": str(r["attack_out_dir"]),
                "backbone": "pixart",
                "surrogate": "S",
                "controlvae": "0",
                "iterations": "5",
                "t_start": str(r["requested_start_step"]),
                "lr": "1e-3",
                "lambda_perc": "0.5",
                "primary_change": str(r["primary_change"]),
                "whitebox_asr": fmt(r["whitebox_asr"]),
                "blackbox_mean_asr": fmt(r["blackbox_mean_asr"]),
                "asr_to_E": fmt(r["asr_to_E"]),
                "asr_to_R": fmt(r["asr_to_R"]),
                "asr_to_D": fmt(r["asr_to_D"]),
                "asr_to_S": fmt(r["asr_to_S"]),
                "lpips": fmt(r["lpips"]),
                "psnr": fmt(r["psnr"]),
                "ssim": fmt(r["ssim"]),
                "fourier_l2_to_real": fmt(r["fourier_l2_to_real"]),
                "residual_fourier_l2_to_real": fmt(r["residual_fourier_l2_to_real"]),
                "attack_mean_psnr": fmt(r["attack_mean_psnr"]),
                "attack_mean_ssim": fmt(r["attack_mean_ssim"]),
                "compared_to": compared_to,
                "new_information": new_information,
                "evidence": str(r["evidence"]),
            }
        )
        out.append(row)
    return out


def append_or_replace_records(new_rows: list[dict[str, str]]) -> None:
    header, rows = load_csv_rows(EXPERIMENT_RECORD_CSV)
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
    write_csv_rows(EXPERIMENT_RECORD_CSV, FIELDNAMES, rows)


def run_one(spec: dict[str, object], force: bool = False) -> dict[str, object]:
    out_dir = run_dir(spec)
    adv_dataroot = output_dataroot(spec)
    summary_path = out_dir / "run_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if adv_dataroot.exists():
        raise FileExistsError(f"Output dataroot already exists: {adv_dataroot}")

    args = attack_args(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    ATTACK_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = attack_log_path()
    start_size = log_path.stat().st_size if log_path.exists() else 0
    write_json(
        out_dir / "run_meta.json",
        {
            "problem": "Compare clean latent, Euler add-noise, and DDIM inversion startpoints on the PixArt main route.",
            "pipeline": [
                "PGD warm start",
                "construct latent startpoint",
                "LAO with self-attn gamma=0.1",
                "DDIM denoise if active_timesteps > 0",
                "export attack / visual / frequency metrics",
            ],
            "startpoint_spec": spec,
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

    export_metrics(spec)
    return collect_summary(spec)


def run_suite(force: bool = False) -> list[dict[str, object]]:
    rows = [run_one(spec, force=force) for spec in RUN_SPECS]
    compute_suite_scores(rows)
    write_json(SUITE_SUMMARY_JSON, rows)
    fieldnames = [
        "run_id",
        "startpoint_family",
        "startpoint_value",
        "requested_start_step",
        "chosen_timestep",
        "sigma_value",
        "whitebox_asr",
        "blackbox_mean_asr",
        "lpips",
        "psnr",
        "ssim",
        "fourier_l2_to_real",
        "residual_fourier_l2_to_real",
        "v_score",
        "s_score",
    ]
    with SUITE_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    SUITE_ANALYSIS_MD.write_text(analyze_suite(rows), encoding="utf-8")
    append_or_replace_records(build_record_rows(rows))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all", help="run_id or 'all'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.run == "all":
        run_suite(force=args.force)
        return 0

    spec = next((x for x in RUN_SPECS if x["run_id"] == args.run), None)
    if spec is None:
        raise KeyError(f"Unknown run_id={args.run!r}")
    row = run_one(spec, force=args.force)
    compute_suite_scores([row])
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
