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
ATTACK_DATASETS_ROOT = DATASET_ROOT / "pixart_alpha" / "attack_datasets"
PIXART_PYTHON = REPO_ROOT / "conda_envs" / "pixart_alpha" / "bin" / "python"
PIXART_MODEL_PATH = REPO_ROOT / "checkpoints" / "hf_models" / "pixart-alpha-xl-2-512x512"
RESULT_ROOT = PIXART_ROOT / "results" / "ensemble_leave_one_out_smoke35_20260327"
RUNS_ROOT = RESULT_ROOT / "runs"
SUITE_SUMMARY_CSV = RESULT_ROOT / "suite_summary.csv"
SUITE_SUMMARY_JSON = RESULT_ROOT / "suite_summary.json"
SUITE_README = RESULT_ROOT / "README.md"
ALL_MODELS = ["E", "R", "D", "S"]

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
}


def build_spec(code: str, seen: list[str]) -> dict[str, object]:
    unseen = [m for m in ALL_MODELS if m not in seen]
    model_name = ",".join([*seen, *unseen])
    primary_surrogate = seen[0]
    run_id = f"pixart_noctrl_smoke35_ens{code}"
    output_name = f"PixArtAlpha_smoke35_adv_noctrl_ens{code}_sur{primary_surrogate}_20260327"
    return {
        "run_id": run_id,
        "code": code,
        "seen": seen,
        "unseen": unseen,
        "primary_surrogate": primary_surrogate,
        "surrogate_label": f"ensemble({','.join(seen)})",
        "model_name": model_name,
        "output_name": output_name,
    }


RUN_SPECS = {
    spec["run_id"]: spec
    for spec in [
        build_spec("ER", ["E", "R"]),
        build_spec("ED", ["E", "D"]),
        build_spec("ES", ["E", "S"]),
        build_spec("RD", ["R", "D"]),
        build_spec("RS", ["R", "S"]),
        build_spec("DS", ["D", "S"]),
        build_spec("ERD", ["E", "R", "D"]),
        build_spec("ERS", ["E", "R", "S"]),
        build_spec("EDS", ["E", "D", "S"]),
        build_spec("RDS", ["R", "D", "S"]),
    ]
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def extract_log_delta(log_path: Path, start_size: int, out_path: Path) -> None:
    if not log_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("rb") as src:
        src.seek(start_size)
        out_path.write_bytes(src.read())


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def output_dataroot(run_id: str) -> Path:
    return ATTACK_DATASETS_ROOT / "smoke35" / str(RUN_SPECS[run_id]["output_name"])


def attack_log_path(run_id: str) -> Path:
    spec = RUN_SPECS[run_id]
    name = str(spec["model_name"]).replace(",", "") + "0"
    surrogate = str(spec["primary_surrogate"])
    return PIXART_ROOT / "exp" / "results" / f"attack_{name}_sur{surrogate}.log"


def attack_args_for_run(run_id: str) -> dict[str, object]:
    spec = RUN_SPECS[run_id]
    args = dict(BASE_ATTACK_ARGS)
    args["model_name"] = str(spec["model_name"])
    args["attack_surrogates"] = ",".join(spec["seen"])
    args["is_encoder"] = 0
    args["pretrained_diffusion_path"] = str(PIXART_MODEL_PATH)
    args["out_dataroot"] = str(output_dataroot(run_id))
    args["intermediate_root"] = str(run_dir(run_id) / "intermediates")
    return args


def write_suite_readme() -> None:
    text = """# Ensemble Surrogate Leave-One-Out Smoke35 Note

This directory stores the smoke35 pilot runs for PixArt no-ControlVAE explicit multi-surrogate ensemble attacks.

- Quantity-1 singleton rows are reused from existing full700 records.
- This batch only runs the 10 unique seen-subset ensembles of size 2 and 3 on `smoke_fixed35`.
- A matched `full700` ensemble matrix still needs to be run later for the formal paper table.

Implementation note:

- `attack_surrogates` changes only the LAO attack loss to an equal-weight average over the selected seen detectors.
- The PGD warm start is kept on the first detector in `model_name`, matching the current PixArt baseline path as closely as possible.
"""
    write_text(SUITE_README, text)


def run_attack(run_id: str) -> None:
    spec = RUN_SPECS[run_id]
    out_dir = run_dir(run_id)
    adv_dataroot = output_dataroot(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (ATTACK_DATASETS_ROOT / "smoke35").mkdir(parents=True, exist_ok=True)
    if adv_dataroot.exists():
        raise FileExistsError(f"Output dataroot already exists: {adv_dataroot}")

    args = attack_args_for_run(run_id)
    log_path = attack_log_path(run_id)
    start_size = log_path.stat().st_size if log_path.exists() else 0

    write_json(
        out_dir / "run_meta.json",
        {
            "run_id": run_id,
            "problem": "Test whether equal-weight multi-surrogate LAO improves held-out transfer on the PixArt no-ControlVAE route.",
            "pipeline": [
                "PixArt transformer backbone",
                "PGD warm start on the first listed detector",
                "LAO latent optimization",
                "equal-weight ensemble attack loss over seen detectors",
                "export detector / visual / frequency metrics",
            ],
            "candidate_modules": {
                "enabled": ["pixart", "equal-weight ensemble surrogate loss"],
                "disabled": ["ControlVAE", "cross-attn regularizer", "self-attn regularizer", "frequency loss"],
            },
            "input_dataset": str(SMOKE35_FAKE_ROOT),
            "clean_dataroot": str(CLEAN_DATAROOT),
            "output_dataset": str(adv_dataroot),
            "output_logs": str(out_dir),
            "backbone": "pixart",
            "implementation": "pixart",
            "seen_surrogates": spec["seen"],
            "unseen_surrogates": spec["unseen"],
            "surrogate_label": spec["surrogate_label"],
            "primary_change": "explicit_multi_surrogate_equal_weight",
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

    extract_log_delta(log_path, start_size, out_dir / log_path.name)
    ensure_symlink(out_dir / "attack_dataroot", adv_dataroot)
    for name in ["manifest_real.csv", "manifest_fake.csv", "manifest_fake_adv.csv", "summary_metrics.json"]:
        copy_text_like(adv_dataroot / name, out_dir / name)


def read_fake_logits(path: Path) -> list[dict[str, str]]:
    rows = read_csv_dicts(path)
    return [row for row in rows if int(row["label"]) == 0]


def asr_from_rows(rows: list[dict[str, str]], target: str, subset: str | None = None) -> float:
    selected = rows
    if subset is not None:
        selected = [row for row in rows if row.get("subset", "") == subset]
    if not selected:
        return float("nan")
    hit = 0
    for row in selected:
        if float(row[f"logit_{target}"]) > 0:
            hit += 1
    return hit / len(selected)


def write_custom_transfer_summary(run_id: str, logits_csv: Path) -> dict[str, object]:
    spec = RUN_SPECS[run_id]
    out_dir = run_dir(run_id)
    fake_rows = read_fake_logits(logits_csv)
    target_asr = {target: asr_from_rows(fake_rows, target) for target in ALL_MODELS}
    seen = list(spec["seen"])
    unseen = list(spec["unseen"])
    whitebox_mean = sum(target_asr[target] for target in seen) / len(seen)
    blackbox_mean = sum(target_asr[target] for target in unseen) / len(unseen)

    transfer_csv = out_dir / "transfer_asr.csv"
    with transfer_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["surrogate", "whitebox_asr", "blackbox_mean_asr"] + [f"asr_to_{target}" for target in ALL_MODELS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "surrogate": str(spec["surrogate_label"]),
            "whitebox_asr": f"{whitebox_mean:.6f}",
            "blackbox_mean_asr": f"{blackbox_mean:.6f}",
        }
        for target in ALL_MODELS:
            row[f"asr_to_{target}"] = f"{target_asr[target]:.6f}"
        writer.writerow(row)

    subsets = sorted({row.get("subset", "") for row in fake_rows if row.get("subset", "")})
    with (out_dir / "transfer_asr_by_subset.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["subset", "target", "asr", "is_seen"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subset in subsets:
            for target in ALL_MODELS:
                writer.writerow(
                    {
                        "subset": subset,
                        "target": target,
                        "asr": f"{asr_from_rows(fake_rows, target, subset=subset):.6f}",
                        "is_seen": "1" if target in seen else "0",
                    }
                )

    return {
        "whitebox_asr": whitebox_mean,
        "blackbox_mean_asr": blackbox_mean,
        "target_asr": target_asr,
    }


def export_metrics(run_id: str) -> None:
    spec = RUN_SPECS[run_id]
    out_dir = run_dir(run_id)
    adv_dataroot = output_dataroot(run_id)

    export_csv = out_dir / "adv_logits.csv"
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
    transfer_stats = write_custom_transfer_summary(run_id, export_csv)
    write_json(out_dir / "transfer_summary.json", transfer_stats)

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
    freq_probe_name = f"{spec['output_name']}_freqprobe_sur{spec['primary_surrogate']}"
    ensure_symlink(freq_root / freq_probe_name, adv_dataroot)
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
            str(spec["primary_surrogate"]),
            "--res",
            "224",
        ],
        cwd=DIFFSR_ROOT,
    )


def collect_summary(run_id: str) -> dict[str, object]:
    spec = RUN_SPECS[run_id]
    out_dir = run_dir(run_id)
    adv_dataroot = output_dataroot(run_id)
    summary_metrics = json.loads((adv_dataroot / "summary_metrics.json").read_text(encoding="utf-8"))
    transfer_row = read_csv_dicts(out_dir / "transfer_asr.csv")[0]
    visual_row = read_csv_dicts(out_dir / "metrics" / "visual" / "visual_quality.csv")[0]
    freq_rows = read_csv_dicts(out_dir / "metrics" / "frequency" / "frequency_metrics.csv")
    freq_row = next(row for row in freq_rows if row["dataset"] == f"sur{spec['primary_surrogate']}")

    target_asr = {target: float(transfer_row[f"asr_to_{target}"]) for target in ALL_MODELS}
    seen = list(spec["seen"])
    unseen = list(spec["unseen"])
    seen_desc = ",".join(seen)
    unseen_desc = ",".join(unseen)
    target_desc = ", ".join(f"{target}={target_asr[target]:.6f}" for target in ALL_MODELS)

    row = {
        "run_id": run_id,
        "backbone": "pixart",
        "implementation": "pixart",
        "surrogate": str(spec["surrogate_label"]),
        "scope": "smoke_fixed35",
        "num_fake_images": 35,
        "primary_change": "explicit_multi_surrogate_equal_weight",
        "seen_surrogates": seen,
        "unseen_surrogates": unseen,
        "attack_out_dir": str(out_dir),
        "adv_dataroot": str(adv_dataroot),
        "whitebox_asr": float(transfer_row["whitebox_asr"]),
        "blackbox_mean_asr": float(transfer_row["blackbox_mean_asr"]),
        "asr_to_E": target_asr["E"],
        "asr_to_R": target_asr["R"],
        "asr_to_D": target_asr["D"],
        "asr_to_S": target_asr["S"],
        "lpips": float(visual_row["lpips_mean"]),
        "psnr": float(visual_row["psnr_mean"]),
        "ssim": float(visual_row["ssim_mean"]),
        "fourier_l2_to_real": float(freq_row["img_logamp_l2_to_real"]),
        "residual_fourier_l2_to_real": float(freq_row["residual_logamp_l2_to_real"]),
        "attack_mean_psnr": float(summary_metrics["mean_psnr"]),
        "attack_mean_ssim": float(summary_metrics["mean_ssim"]),
        "controlvae": "off",
        "compared_to": "|".join(f"pixart_noctrl_700_sur{x}" for x in seen),
        "new_information": (
            f"Equal-weight ensemble over seen={seen_desc} gives seen-mean ASR={float(transfer_row['whitebox_asr']):.6f} "
            f"and unseen-mean ASR={float(transfer_row['blackbox_mean_asr']):.6f}; per-target ASR: {target_desc}. "
            f"This smoke35 pilot is compared mainly against singleton anchors for the same seen detectors, while unseen={unseen_desc}."
        ),
        "evidence": str(out_dir / "run_summary.json"),
    }
    write_json(out_dir / "run_summary.json", row)
    return row


def summarize_completed_runs() -> None:
    rows = []
    for run_id in RUN_SPECS:
        summary_path = run_dir(run_id) / "run_summary.json"
        if summary_path.exists():
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))

    if not rows:
        return

    fieldnames = [
        "run_id",
        "surrogate",
        "scope",
        "whitebox_asr",
        "blackbox_mean_asr",
        "asr_to_E",
        "asr_to_R",
        "asr_to_D",
        "asr_to_S",
        "psnr",
        "ssim",
        "lpips",
        "fourier_l2_to_real",
        "residual_fourier_l2_to_real",
        "attack_out_dir",
        "adv_dataroot",
    ]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with SUITE_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(SUITE_SUMMARY_JSON, {"rows": rows, "summary_csv": str(SUITE_SUMMARY_CSV)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", choices=sorted(RUN_SPECS.keys()), help="Run one or more specific run ids")
    ap.add_argument("--skip_attack", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    write_suite_readme()
    run_ids = args.run if args.run else list(RUN_SPECS.keys())
    for run_id in run_ids:
        if not args.skip_attack:
            run_attack(run_id)
        if not args.skip_eval:
            export_metrics(run_id)
            collect_summary(run_id)
    summarize_completed_runs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
