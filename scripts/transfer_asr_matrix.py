#!/usr/bin/env python3
"""
Compute transfer ASR matrix from per-image logits CSVs.

Each input CSV is expected to be produced by `scripts/eval_detectors.py --out_csv ...`
and contain columns:
  - label (0=fake, 1=real)
  - subset
  - logit_E, logit_R, logit_D, logit_S  (or a subset of these)

ASR definition used here (fake->real evasion):
  ASR(target) = mean( logit_target > 0 ) over fake images (label==0)

Usage:
  python scripts/transfer_asr_matrix.py \
    --run E=exp/results/adv_logits_surE.csv \
    --run R=exp/results/adv_logits_surR.csv \
    --run D=exp/results/adv_logits_surD.csv \
    --run S=exp/results/adv_logits_surS.csv \
    --out_csv exp/results/transfer_asr.csv

Optionally:
  --by_subset_csv exp/results/transfer_asr_by_subset.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


ALL_MODELS = ["E", "R", "D", "S"]


def parse_run_arg(s: str) -> Tuple[str, Path]:
    if "=" not in s:
        raise ValueError(f"Invalid --run {s!r}. Expected like E=path.csv")
    k, v = s.split("=", 1)
    k = k.strip()
    if k not in ALL_MODELS:
        raise ValueError(f"Invalid surrogate {k!r}. Must be one of {ALL_MODELS}")
    p = Path(v.strip())
    return k, p


def read_logits_csv(p: Path) -> List[Dict[str, str]]:
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"Empty CSV: {p}")
    if "label" not in rows[0]:
        raise ValueError(f"CSV missing 'label' column: {p}")
    return rows


def asr_over_rows(rows: List[Dict[str, str]], target: str, subset: str | None = None) -> float:
    col = f"logit_{target}"
    num = 0
    den = 0
    for row in rows:
        if int(row["label"]) != 0:
            continue
        if subset is not None and row.get("subset", "") != subset:
            continue
        if col not in row:
            raise ValueError(f"Missing column {col!r} in input CSV")
        logit = float(row[col])
        den += 1
        if logit > 0:
            num += 1
    if den == 0:
        return float("nan")
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Surrogate=path.csv (repeatable)")
    ap.add_argument("--out_csv", required=True, help="Output matrix CSV path")
    ap.add_argument("--by_subset_csv", default="", help="Optional output CSV with per-subset ASR matrix")
    args = ap.parse_args()

    runs: Dict[str, Path] = {}
    for s in args.run:
        k, p = parse_run_arg(s)
        runs[k] = p

    # Read all rows.
    data: Dict[str, List[Dict[str, str]]] = {k: read_logits_csv(p) for k, p in runs.items()}

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["surrogate"] + [f"asr_to_{t}" for t in ALL_MODELS if t in ALL_MODELS])
        for sur in ALL_MODELS:
            if sur not in data:
                continue
            row = [sur]
            for tgt in ALL_MODELS:
                if tgt == sur:
                    row.append("")  # diagonal not counted in 4*3 transfer values
                else:
                    row.append(f"{asr_over_rows(data[sur], tgt):.6f}")
            w.writerow(row)

    print(f"Wrote transfer ASR matrix: {outp}")

    if args.by_subset_csv:
        # Collect subset names from all runs.
        subsets = set()
        for rows in data.values():
            for r in rows:
                if int(r["label"]) == 0:
                    subsets.add(r.get("subset", "unknown"))
        subsets = set(s for s in subsets if s) or {"unknown"}
        subset_list = sorted(subsets)

        outp2 = Path(args.by_subset_csv)
        outp2.parent.mkdir(parents=True, exist_ok=True)
        with outp2.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subset", "surrogate", "target", "asr"])
            for subset in subset_list:
                for sur in ALL_MODELS:
                    if sur not in data:
                        continue
                    for tgt in ALL_MODELS:
                        if tgt == sur:
                            continue
                        w.writerow([subset, sur, tgt, f"{asr_over_rows(data[sur], tgt, subset=subset):.6f}"])
        print(f"Wrote per-subset transfer ASR: {outp2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

