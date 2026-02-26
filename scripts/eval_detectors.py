#!/usr/bin/env python3
"""
Evaluate E/R/D/S detectors on a dataroot that contains:
  dataroot/real/*   (label=1)
  dataroot/fake/*   (label=0; can contain subset subfolders)

Outputs overall metrics and per-fake-subset metrics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import other_attacks


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort(key=lambda q: q.as_posix())
    return files


def infer_label(p: Path) -> int:
    parts = [x.lower() for x in p.parts]
    has_fake = any(x == "fake" or x.startswith("fake") for x in parts)
    has_real = any(x == "real" or x.startswith("real") for x in parts)
    if has_fake and not has_real:
        return 0
    if has_real and not has_fake:
        return 1
    raise ValueError(f"Can't infer label from {p}")


def infer_fake_subset(p: Path) -> str:
    # subset is the folder right under `fake/` if present; else try filename token
    parts = list(p.parts)
    for i, x in enumerate(parts):
        if x.lower() == "fake" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if Path(nxt).suffix == "":
                return nxt.lower()
    stem = p.stem
    toks = stem.split("_")
    if len(toks) >= 2 and toks[0].isdigit():
        return toks[1].lower()
    if len(toks) >= 1 and not toks[0].isdigit():
        return toks[0].lower()
    return "unknown"


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve_np(y_true, score)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(np.max(tpr[ok]))


@dataclass
class Metrics:
    n: int
    acc: float
    bacc: float
    auc: float
    ap: float
    tpr_fpr1: float
    tpr_fpr5: float


def roc_curve_np(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal ROC curve implementation (pos_label=1).
    Returns fpr, tpr, thresholds (descending).
    """
    y = y_true.astype(np.int64)
    s = score.astype(np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]

    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        raise ValueError("ROC is undefined when there are no positive or no negative samples.")

    distinct = np.where(np.diff(s) != 0)[0]
    thr_idx = np.r_[distinct, len(s) - 1]
    tps = np.cumsum(y == 1)[thr_idx]
    fps = (thr_idx + 1) - tps

    tpr = tps / P
    fpr = fps / N
    thresholds = s[thr_idx]

    # Add origin
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[thresholds[0] + 1e-12, thresholds]
    return fpr, tpr, thresholds


def roc_auc_np(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve_np(y_true, score)
    return float(np.trapz(tpr, fpr))


def average_precision_np(y_true: np.ndarray, score: np.ndarray) -> float:
    """
    Average precision (area under precision-recall curve), pos_label=1.
    """
    y = y_true.astype(np.int64)
    s = score.astype(np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]

    P = int(np.sum(y == 1))
    if P == 0:
        raise ValueError("AP is undefined when there are no positive samples.")

    distinct = np.where(np.diff(s) != 0)[0]
    thr_idx = np.r_[distinct, len(s) - 1]
    tps = np.cumsum(y == 1)[thr_idx]
    fps = (thr_idx + 1) - tps

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / P

    # AP = sum over recall steps of precision * delta_recall
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = np.sum(precision * (recall - recall_prev))
    return float(ap)


def compute_metrics(y_true: np.ndarray, score: np.ndarray) -> Metrics:
    y_pred = (score > 0).astype(np.int64)  # OURS wrapper uses [0, logit]; threshold at 0 matches argmax
    acc = float(np.mean(y_pred == y_true))
    # balanced accuracy = (TPR + TNR) / 2
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    bacc = float((tpr + tnr) / 2.0)
    auc = roc_auc_np(y_true, score)
    ap = average_precision_np(y_true, score)
    return Metrics(
        n=int(len(y_true)),
        acc=acc,
        bacc=bacc,
        auc=auc,
        ap=ap,
        tpr_fpr1=tpr_at_fpr(y_true, score, 0.01),
        tpr_fpr5=tpr_at_fpr(y_true, score, 0.05),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, help="Folder with real/ and fake/")
    ap.add_argument("--models", default="E,R,D,S", help="Comma-separated list from {E,R,D,S}")
    ap.add_argument("--res", type=int, default=224, help="Resize to res x res")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out_csv", default="", help="Optional CSV path to write per-image scores")
    args = ap.parse_args()

    dataroot = Path(args.dataroot)
    files = iter_images(dataroot)
    if not files:
        raise FileNotFoundError(f"No images under {dataroot}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    nets = {m: other_attacks.model_selection(m, device=device).eval() for m in models}
    for net in nets.values():
        net.requires_grad_(False)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose(
        [
            transforms.Resize((args.res, args.res), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    y_true = []
    subset = []
    # store per-model logits for class=1 (real)
    scores: Dict[str, List[float]] = {m: [] for m in models}
    paths: List[str] = []

    # `model_selection` already placed nets on `device`.

    # simple batching
    for i in range(0, len(files), args.batch):
        batch_files = files[i : i + args.batch]
        xs = []
        for p in batch_files:
            img = Image.open(p).convert("RGB")
            xs.append(tfm(img))
        x = torch.stack(xs, dim=0).to(device)

        with torch.no_grad():
            for m, net in nets.items():
                logits = net(x)  # [B,2], where logits[:,1] is the learned logit
                # Some models may output shape [B,1] (single logit). Others output [B,2] ([0, z]).
                if logits.ndim != 2:
                    raise ValueError(f"Unexpected logits shape for {m}: {tuple(logits.shape)}")
                if logits.shape[1] == 2:
                    z = logits[:, 1]
                elif logits.shape[1] == 1:
                    z = logits[:, 0]
                else:
                    raise ValueError(f"Unexpected logits shape for {m}: {tuple(logits.shape)}")
                scores[m].extend(z.detach().cpu().numpy().astype(np.float64).tolist())

        for p in batch_files:
            y_true.append(infer_label(p))
            subset.append(infer_fake_subset(p) if infer_label(p) == 0 else "real")
            paths.append(str(p))

    y_true_np = np.asarray(y_true, dtype=np.int64)

    print(f"files: {len(files)} (real={(y_true_np==1).sum()}, fake={(y_true_np==0).sum()})")
    print()

    rows = []
    for m in models:
        score_np = np.asarray(scores[m], dtype=np.float64)
        met = compute_metrics(y_true_np, score_np)
        print(
            f"[{m}] n={met.n} acc={met.acc:.4f} bacc={met.bacc:.4f} "
            f"auc={met.auc:.4f} ap={met.ap:.4f} "
            f"tpr@fpr1%={met.tpr_fpr1:.4f} tpr@fpr5%={met.tpr_fpr5:.4f}"
        )
        rows.append((m, met))

    # Per fake subset (real excluded) -- helpful for GenImage heterogeneity.
    print()
    print("Per-subset (fake only):")
    subset_arr = np.asarray(subset)
    is_fake = y_true_np == 0
    for m in models:
        score_np = np.asarray(scores[m], dtype=np.float64)
        for s in sorted(set(subset_arr[is_fake])):
            idx = np.where((subset_arr == s) & is_fake)[0]
            # y_true is all 0 here; AUC/AP undefined. We report mean score and ASR-like rate.
            # ASR: fraction predicted as real (score>0).
            asr = float(np.mean(score_np[idx] > 0))
            mean_score = float(np.mean(score_np[idx]))
            print(f"[{m}] subset={s:>12s} n={len(idx):4d} asr(fake->real)={asr:.4f} mean_logit={mean_score:.4f}")
        print()

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "subset"] + [f"logit_{m}" for m in models])
            for i, p in enumerate(paths):
                w.writerow([p, int(y_true_np[i]), subset[i]] + [scores[m][i] for m in models])
        print()
        print(f"Wrote per-image logits: {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
