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

SUITE_ROOT = PIXART_ROOT / "results" / "full_700_20260323"
RUNS_ROOT = SUITE_ROOT / "runs"
SUMMARY_CSV = SUITE_ROOT / "suite_summary.csv"
SUMMARY_JSON = SUITE_ROOT / "suite_summary.json"
SUMMARY_MD = SUITE_ROOT / "suite_summary.md"
MANIFEST_CSV = SUITE_ROOT / "run_manifest.csv"
SUITE_SPEC_JSON = SUITE_ROOT / "suite_spec.json"
SUITE_SPEC_MD = SUITE_ROOT / "suite_spec.md"

EVAL_ROOT = DIFFSR_ROOT / "results" / "PixArtAlpha_noctrlvae"

PIXART_MODEL_PATH = REPO_ROOT / "checkpoints" / "hf_models" / "pixart-alpha-xl-2-512x512"
PIXART_PYTHON = REPO_ROOT / "conda_envs" / "pixart_alpha" / "bin" / "python"

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
}

RUN_SPECS = {
    "pixart_noctrl_700_surE": {
        "backbone": "pixart",
        "surrogate": "E",
        "controlvae": 0,
        "model_name": "E,R,D,S",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "diffattack_700_surE|sd_ctrl_700_surE",
    },
    "pixart_noctrl_700_surR": {
        "backbone": "pixart",
        "surrogate": "R",
        "controlvae": 0,
        "model_name": "R,E,D,S",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "diffattack_700_surR|sd_ctrl_700_surR",
    },
    "pixart_noctrl_700_surD": {
        "backbone": "pixart",
        "surrogate": "D",
        "controlvae": 0,
        "model_name": "D,E,R,S",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "diffattack_700_surD|sd_ctrl_700_surD",
    },
    "pixart_noctrl_700_surS": {
        "backbone": "pixart",
        "surrogate": "S",
        "controlvae": 0,
        "model_name": "S,E,R,D",
        "pretrained_diffusion_path": str(PIXART_MODEL_PATH),
        "compared_to": "diffattack_700_surS|sd_ctrl_700_surS|sd_noctrl_700_surS|sr_noctrl_700_surS",
    },
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, start=link_path.parent)
    os.symlink(rel, link_path)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_text_like(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def extract_log_delta(log_path: Path, start_size: int, out_path: Path) -> None:
    if not log_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("rb") as src:
        src.seek(start_size)
        out_path.write_bytes(src.read())


def attack_log_path(spec: dict[str, object]) -> Path:
    model_name = str(spec["model_name"]).split(",")
    surrogate = str(spec["surrogate"])
    is_encoder = int(spec["controlvae"])
    name = "".join(model_name) + str(is_encoder)
    return PIXART_ROOT / "exp" / "results" / f"attack_{name}_sur{surrogate}.log"


def run_output_name(run_id: str) -> str:
    surrogate = RUN_SPECS[run_id]["surrogate"]
    return f"PixArtAlpha_genimage_eval_1400_adv_noctrlvae_sur{surrogate}"


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


def run_attack(run_id: str) -> None:
    spec = RUN_SPECS[run_id]
    out_dir = run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    ATTACK_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    adv_dataroot = run_output_dataroot(run_id)
    if adv_dataroot.exists():
        raise FileExistsError(f"Output dataroot already exists: {adv_dataroot}")

    args = attack_args_for_run(run_id)
    log_path = attack_log_path(spec)
    start_size = log_path.stat().st_size if log_path.exists() else 0

    run_meta = {
        "run_id": run_id,
        "problem": "Run PixArt-Alpha no-ControlVAE full GenImage-700 attack for one surrogate and export evaluation-ready dataroots.",
        "pipeline": [
            "load PixArt-Alpha backbone",
            "encode image with shared VAE",
            "DDIM inversion",
            "latent optimization / LAO",
            "DDIM denoise",
            "decode with VAE",
            "export full eval dataroot",
            "run detector / visual / frequency evaluation",
        ],
        "candidate_modules": {
            "enabled": ["PixArtAlphaPipeline.transformer", "vae", "tokenizer", "text_encoder", "detectors E/R/D/S"],
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
        "input_dataset": str(FULL_FAKE_ROOT),
        "clean_dataroot": str(CLEAN_DATAROOT),
        "output_dataset": str(adv_dataroot),
        "output_logs": str(out_dir),
        "output_weights": "none",
        "backbone": spec["backbone"],
        "surrogate": spec["surrogate"],
        "controlvae": spec["controlvae"],
        "compared_to": spec["compared_to"],
        "hyperparams": {
            "diffusion_steps": args["diffusion_steps"],
            "start_step": args["start_step"],
            "iterations": args["iterations"],
            "lr": args["lr"],
            "guidance": args["guidance"],
            "lambda_lpips": args["lambda_lpips"],
            "eps": args["eps"],
            "pgd_steps": args["pgd_steps"],
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

    ensure_symlink(out_dir / "attack_dataroot", adv_dataroot)
    extract_log_delta(log_path, start_size, out_dir / attack_log_path(spec).name)

    for name in ["manifest_real.csv", "manifest_fake.csv", "manifest_fake_adv.csv", "summary_metrics.json", "meta.txt"]:
        copy_text_like(adv_dataroot / name, out_dir / name)


def evaluate_run(run_id: str) -> None:
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
    summary_metrics = json.loads((run_output_dataroot(run_id) / "summary_metrics.json").read_text(encoding="utf-8"))
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


def write_suite_spec(run_ids: list[str]) -> None:
    surrogates = [str(RUN_SPECS[run_id]["surrogate"]) for run_id in run_ids]
    spec = {
        "problem": "Evaluate whether PixArt-Alpha can serve as the no-ControlVAE attack backbone for full GenImage-700 fake-image evasion across surrogates E/R/D/S while preserving visual and frequency stealth.",
        "pipeline": [
            "load PixArt-Alpha backbone",
            "encode clean fake image into latent",
            "run DDIM inversion",
            "apply PGD warm start + latent optimization",
            "run DDIM denoising and decode adversarial image",
            "export full evaluation dataroot with clean real + adversarial fake",
            "evaluate detectors, visual quality, and frequency statistics",
        ],
        "candidate_modules": {
            "enabled": ["PixArtAlphaPipeline.transformer", "vae", "tokenizer", "text_encoder", "detectors E/R/D/S"],
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
        "input_dataset": {
            "clean_dataroot": str(CLEAN_DATAROOT),
            "attack_source_fake": str(FULL_FAKE_ROOT),
        },
        "output_dataset_root": str(ATTACK_DATASETS_ROOT),
        "output_eval_root": str(EVAL_ROOT),
        "output_logs_root": str(SUITE_ROOT),
        "output_weights": "none",
        "run_ids": run_ids,
        "surrogates": surrogates,
    }
    write_json(SUITE_SPEC_JSON, spec)
    lines = [
        "# PixArt-Alpha full-700 suite spec",
        "",
        "## Problem",
        f"- {spec['problem']}",
        "",
        "## Pipeline",
    ]
    lines.extend([f"- {item}" for item in spec["pipeline"]])
    lines.extend(
        [
            "",
            "## Candidate Modules",
            f"- enabled: {', '.join(spec['candidate_modules']['enabled'])}",
            f"- disabled: {', '.join(spec['candidate_modules']['disabled'])}",
            "",
            "## Metrics",
        ]
    )
    lines.extend([f"- {item}" for item in spec["metrics"]])
    lines.extend(
        [
            "",
            "## IO",
            f"- input clean dataroot: `{CLEAN_DATAROOT}`",
            f"- input fake source: `{FULL_FAKE_ROOT}`",
            f"- output datasets: `{ATTACK_DATASETS_ROOT}`",
            f"- output evaluation: `{EVAL_ROOT}`",
            f"- output logs: `{SUITE_ROOT}`",
            "- output weights: none",
        ]
    )
    SUITE_SPEC_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(run_ids: list[str]) -> None:
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "surrogate", "adv_dataroot", "run_dir", "run_summary", "status"],
        )
        writer.writeheader()
        for run_id in run_ids:
            writer.writerow(
                {
                    "run_id": run_id,
                    "surrogate": RUN_SPECS[run_id]["surrogate"],
                    "adv_dataroot": str(run_output_dataroot(run_id)),
                    "run_dir": str(run_dir(run_id)),
                    "run_summary": str(run_dir(run_id) / "run_summary.json"),
                    "status": "complete" if (run_dir(run_id) / "run_summary.json").exists() else "missing",
                }
            )


def summarize_suite(run_ids: list[str]) -> list[dict[str, object]]:
    rows = [json.loads((run_dir(run_id) / "run_summary.json").read_text(encoding="utf-8")) for run_id in run_ids]

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
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_whitebox = max(rows, key=lambda row: float(row["whitebox_asr"]))
    best_blackbox = max(rows, key=lambda row: float(row["blackbox_mean_asr"]))
    best_psnr = max(rows, key=lambda row: float(row["psnr"]))
    best_fourier = min(rows, key=lambda row: float(row["fourier_l2_to_real"]))

    md_lines = [
        "# PixArt-Alpha full GenImage-700 suite (2026-03-23)",
        "",
        "## Problem",
        "- Evaluate the full 700-fake PixArt-Alpha no-ControlVAE attack matrix across surrogates E/R/D/S.",
        "",
        "## Outputs",
        f"- Suite root: `{SUITE_ROOT}`",
        f"- Dataset root: `{ATTACK_DATASETS_ROOT}`",
        f"- Batch eval root: `{EVAL_ROOT}`",
        "",
        "## Best rows",
        f"- Whitebox ASR: `{best_whitebox['run_id']}` = `{best_whitebox['whitebox_asr']:.6f}`",
        f"- Blackbox mean ASR: `{best_blackbox['run_id']}` = `{best_blackbox['blackbox_mean_asr']:.6f}`",
        f"- PSNR: `{best_psnr['run_id']}` = `{best_psnr['psnr']:.6f}`",
        f"- Fourier L2 to real: `{best_fourier['run_id']}` = `{best_fourier['fourier_l2_to_real']:.6f}`",
        "",
        "## Summary CSV",
        f"- `{SUMMARY_CSV}`",
    ]
    SUMMARY_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    write_json(
        SUMMARY_JSON,
        {
            "manifest_csv": str(MANIFEST_CSV),
            "summary_csv": str(SUMMARY_CSV),
            "summary_md": str(SUMMARY_MD),
            "rows": rows,
            "best_whitebox": best_whitebox,
            "best_blackbox": best_blackbox,
            "best_psnr": best_psnr,
            "best_fourier": best_fourier,
        },
    )
    write_manifest(run_ids)
    return rows


def run_batch_evaluation(run_ids: list[str]) -> None:
    surrogates = ",".join(str(RUN_SPECS[run_id]["surrogate"]) for run_id in run_ids)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "export_detector_logits.py"),
            "--dataroot",
            str(CLEAN_DATAROOT),
            "--models",
            "E,R,D,S",
            "--batch",
            "32",
            "--out_csv",
            str(EVAL_ROOT / "clean_logits.csv"),
        ],
        cwd=DIFFSR_ROOT,
    )

    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "run_batch_eval.py"),
            "--root",
            str(ATTACK_DATASETS_ROOT),
            "--out_dir",
            str(EVAL_ROOT),
            "--models",
            "E,R,D,S",
            "--surrogates",
            surrogates,
            "--batch",
            "32",
        ],
        cwd=DIFFSR_ROOT,
    )

    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "eval_visual_quality.py"),
            "--clean_dataroot",
            str(CLEAN_DATAROOT),
            "--root",
            str(ATTACK_DATASETS_ROOT),
            "--out_dir",
            str(EVAL_ROOT),
        ],
        cwd=DIFFSR_ROOT,
    )

    run_cmd(
        [
            str(PIXART_PYTHON),
            str(DIFFSR_ROOT / "evaluation" / "eval_frequency_metrics.py"),
            "--clean_dataroot",
            str(CLEAN_DATAROOT),
            "--root",
            str(ATTACK_DATASETS_ROOT),
            "--out_dir",
            str(EVAL_ROOT),
            "--surrogates",
            surrogates,
            "--res",
            "224",
        ],
        cwd=DIFFSR_ROOT,
    )

    meta = {
        "problem": "Full GenImage-700 PixArt-Alpha no-ControlVAE four-surrogate evaluation.",
        "pipeline": "attack dataroot generation -> detector logits -> transfer ASR -> visual quality -> frequency metrics",
        "suite_root": str(SUITE_ROOT),
        "dataset_root": str(ATTACK_DATASETS_ROOT),
        "runs": run_ids,
        "surrogates": [str(RUN_SPECS[run_id]["surrogate"]) for run_id in run_ids],
        "weights": "none",
    }
    write_json(EVAL_ROOT / "meta.json", meta)


def execute_run(run_id: str) -> None:
    print(f"=== {run_id}: attack ===", flush=True)
    run_attack(run_id)
    print(f"=== {run_id}: eval ===", flush=True)
    evaluate_run(run_id)
    collect_run_summary(run_id)
    print(f"=== {run_id}: done ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "eval", "summary"])
    ap.add_argument("--runs", default="", help="Comma-separated run ids. Default: all suite runs.")
    args = ap.parse_args()

    run_ids = [x.strip() for x in args.runs.split(",") if x.strip()]
    if not run_ids:
        run_ids = list(RUN_SPECS.keys())

    if args.action == "run":
        write_suite_spec(run_ids)
        for run_id in run_ids:
            execute_run(run_id)
        run_batch_evaluation(run_ids)
        summarize_suite(run_ids)
        return 0

    if args.action == "eval":
        write_suite_spec(run_ids)
        for run_id in run_ids:
            evaluate_run(run_id)
            collect_run_summary(run_id)
        run_batch_evaluation(run_ids)
        summarize_suite(run_ids)
        return 0

    write_suite_spec(run_ids)
    summarize_suite(run_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
