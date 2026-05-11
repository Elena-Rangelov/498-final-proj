"""
Analyze 05_train.py output logs and CV checkpoints.

Features
--------
1) Parse a stdout log captured from `python 05_train.py ... > train_log.txt`
   and plot training loss vs epoch (optionally with one line per fold).
2) Load each fold's `best.pt` checkpoint, re-create the CV splits, and compute:
     - per-class F1 (H/E/C)
     - per-class SOV'99 (segment overlap score) (H/E/C)
   on either the fold validation split (default) or the shared held-out test set.

Example
-------
python3 07_analyze_training_log.py \
  --log runs/exp1_gpu/train_log.txt \
  --runs_dir runs/exp1_gpu \
  --out_dir runs/exp1_gpu/plots
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_model_mod = _load("model", "03_model.py")
SSPredictor = _model_mod.SSPredictor
SS3_CHARS = _model_mod.SS3_CHARS
SS3_TO_IDX = _model_mod.SS3_TO_IDX


EPOCH_RE = re.compile(
    r"^(?:\[(?:fold\s+(?P<fold>\d+)/(?P<n_folds>\d+))\]\s+)?"
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+train_loss=(?P<train_loss>[0-9.]+)\s+\|\s+"
    r"val_q3=(?P<val_q3>[0-9.]+)"
)


def parse_training_log(log_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = EPOCH_RE.match(line)
            if not m:
                continue
            fold = int(m.group("fold") or 1)
            n_folds = int(m.group("n_folds") or 1)
            rows.append(
                {
                    "fold": fold,
                    "n_folds": n_folds,
                    "epoch": int(m.group("epoch")),
                    "train_loss": float(m.group("train_loss")),
                    "val_q3": float(m.group("val_q3")),
                }
            )

    if not rows:
        raise SystemExit(
            "No epoch lines found in log. Expected lines like:\n"
            "[fold 1/5] Epoch   1 | train_loss=... | val_q3=..."
        )

    df = pd.DataFrame(rows).sort_values(["fold", "epoch"]).reset_index(drop=True)
    return df


def cv_splits(
    embedding_dir: Path,
    test_frac: float,
    n_folds: int,
    seed: int,
) -> tuple[list[str], list[tuple[list[str], list[str]]]]:
    """Mirror 05_train.py: fixed test set + KFold splits over the remainder."""
    ids = sorted(p.stem for p in embedding_dir.glob("*.pt"))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_test = int(len(ids) * test_frac)
    test_ids = ids[:n_test]
    cv_ids = np.array(ids[n_test:])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [
        (cv_ids[train_idx].tolist(), cv_ids[val_idx].tolist())
        for train_idx, val_idx in kf.split(cv_ids)
    ]
    return test_ids, folds


def _segments(seq: str, state: str) -> list[tuple[int, int]]:
    """Return maximal half-open [start, end) segments with seq == state."""
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for i, c in enumerate(seq):
        if c == state and start is None:
            start = i
        elif c != state and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(seq)))
    return segs


@dataclass(frozen=True)
class SovParts:
    numer: float
    denom: float


def sov99_parts(true_seq: str, pred_seq: str, states: tuple[str, ...] = tuple(SS3_CHARS)) -> dict[str, SovParts]:
    """
    Compute additive parts (numerator/denominator) for SOV'99 (Zemla et al., 1999).

    This follows the definition used in many SSP papers:
      - S(i)  : set of all overlapping segment pairs (s1, s2) in state i
      - S'(i) : segments s1 in state i with no overlapping s2
      - N(i)  : sum_{S(i)} len(s1) + sum_{S'(i)} len(s1)
      - delta : min(maxov-minov, minov, int(len(s1)/2), int(len(s2)/2))
    """
    if len(true_seq) != len(pred_seq):
        raise ValueError("true_seq and pred_seq must have the same length")

    out: dict[str, SovParts] = {}
    for state in states:
        true_segs = _segments(true_seq, state)
        pred_segs = _segments(pred_seq, state)

        numer = 0.0
        denom = 0.0

        for a1, b1 in true_segs:
            len1 = b1 - a1
            overlaps: list[tuple[int, int]] = []
            for a2, b2 in pred_segs:
                if min(b1, b2) - max(a1, a2) > 0:
                    overlaps.append((a2, b2))

            if not overlaps:
                denom += len1
                continue

            for a2, b2 in overlaps:
                len2 = b2 - a2
                minov = max(0, min(b1, b2) - max(a1, a2))
                maxov = max(b1, b2) - min(a1, a2)
                delta = min(maxov - minov, minov, len1 // 2, len2 // 2)
                numer += ((minov + delta) / maxov) * len1
                denom += len1

        out[state] = SovParts(numer=numer, denom=denom)

    return out


def _confusion_to_prf1(conf: np.ndarray) -> dict[str, dict[str, float]]:
    per_class: dict[str, dict[str, float]] = {}
    for c, name in enumerate(SS3_CHARS):
        tp = float(conf[c, c])
        fp = float(conf[:, c].sum() - conf[c, c])
        fn = float(conf[c, :].sum() - conf[c, c])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    macro = {
        "precision": float(np.mean([per_class[c]["precision"] for c in SS3_CHARS])),
        "recall": float(np.mean([per_class[c]["recall"] for c in SS3_CHARS])),
        "f1": float(np.mean([per_class[c]["f1"] for c in SS3_CHARS])),
    }
    per_class["macro"] = macro
    return per_class


def evaluate_checkpoint_on_ids(
    checkpoint_path: Path,
    embedding_dir: Path,
    eval_ids: list[str],
    device: str,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    model = SSPredictor(
        esm_dim=int(args["esm_dim"]),
        d_model=int(args["d_model"]),
        n_heads=int(args["n_heads"]),
        n_layers=int(args["n_layers"]),
        d_ff=int(args["d_ff"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    conf = np.zeros((3, 3), dtype=np.int64)
    n_total = 0
    n_correct = 0

    sov_nums: dict[str, float] = {s: 0.0 for s in SS3_CHARS}
    sov_denoms: dict[str, float] = {s: 0.0 for s in SS3_CHARS}

    with torch.no_grad():
        for uid in eval_ids:
            path = embedding_dir / f"{uid}.pt"
            if not path.exists():
                continue
            item = torch.load(path, weights_only=False)
            true_seq: str = item["ss3"]
            emb = item["embedding"].float().to(device)  # (L, d)
            L = emb.shape[0]
            mask = torch.ones(1, L, dtype=torch.bool, device=device)

            logits = model(emb.unsqueeze(0), mask)
            pred_idx = logits.argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int64)
            true_idx = np.fromiter((SS3_TO_IDX[c] for c in true_seq), count=L, dtype=np.int64)

            n_total += L
            n_correct += int((pred_idx == true_idx).sum())
            np.add.at(conf, (true_idx, pred_idx), 1)

            pred_seq = "".join(SS3_CHARS[i] for i in pred_idx.tolist())
            parts = sov99_parts(true_seq, pred_seq)
            for s, p in parts.items():
                sov_nums[s] += p.numer
                sov_denoms[s] += p.denom

    q3 = n_correct / max(n_total, 1)
    prf1 = _confusion_to_prf1(conf)

    sov_per_class = {
        s: (100.0 * sov_nums[s] / sov_denoms[s]) if sov_denoms[s] > 0 else 0.0 for s in SS3_CHARS
    }
    sov_n_total = float(sum(sov_nums.values()))
    sov_d_total = float(sum(sov_denoms.values()))
    sov_overall = (100.0 * sov_n_total / sov_d_total) if sov_d_total > 0 else 0.0

    return {
        "q3": q3,
        "confusion": conf.tolist(),
        "prf1": prf1,
        "sov": {"overall": sov_overall, "per_class": sov_per_class},
    }


def plot_loss(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    hue = "fold" if df["fold"].nunique() > 1 else None
    sns.lineplot(
        data=df,
        x="epoch",
        y="train_loss",
        hue=hue,
        marker="o",
        ax=ax,
    )
    ax.set_title("Training loss per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss (CrossEntropy)")
    if hue is None and ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_fold_metrics(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    f1 = metrics_df[metrics_df["metric"].eq("f1")].copy()
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=f1, x="fold", y="value", hue="class", ax=ax_f1)
    ax_f1.set_title("F1 by class (H/E/C) per fold")
    ax_f1.set_xlabel("Fold")
    ax_f1.set_ylabel("F1")
    ax_f1.set_ylim(0.0, 1.0)
    ax_f1.legend(title="Class", loc="lower right")
    fig_f1.tight_layout()
    fig_f1.savefig(out_dir / "fold_f1.png", dpi=200)
    plt.close(fig_f1)

    sov = metrics_df[metrics_df["metric"].eq("sov")].copy()
    fig_sov, ax_sov = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sov, x="fold", y="value", hue="class", ax=ax_sov)
    ax_sov.set_title("SOV'99 by class (H/E/C) per fold")
    ax_sov.set_xlabel("Fold")
    ax_sov.set_ylabel("SOV (%)")
    ax_sov.set_ylim(0.0, 100.0)
    ax_sov.legend(title="Class", loc="lower right")
    fig_sov.tight_layout()
    fig_sov.savefig(out_dir / "fold_sov.png", dpi=200)
    plt.close(fig_sov)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True, help="Text file containing 05_train.py stdout.")
    ap.add_argument(
        "--runs_dir",
        type=Path,
        default=None,
        help="Directory with fold_*/best.pt (enables PRF1 + SOV evaluation).",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("runs/plots"), help="Output directory for plots/CSVs.")
    ap.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which data to score each fold on: fold validation (val) or shared held-out test (test).",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda / cpu (default: cuda if available).",
    )
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"--log not found: {args.log}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_training_log(args.log)
    plot_loss(df, out_dir / "loss_curve.png")
    df.to_csv(out_dir / "loss_curve.csv", index=False)

    if args.runs_dir is None:
        return

    fold_dirs = sorted([p for p in args.runs_dir.glob("fold_*") if p.is_dir()], key=lambda p: p.name)
    if not fold_dirs:
        raise SystemExit(f"No fold_* directories found under: {args.runs_dir}")

    first_ckpt = fold_dirs[0] / "best.pt"
    if not first_ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {first_ckpt}")

    ckpt0 = torch.load(first_ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt0["args"]
    embedding_dir = Path(train_args["embedding_dir"])
    if not embedding_dir.is_absolute():
        embedding_dir = (Path(__file__).parent / embedding_dir).resolve()
    if not embedding_dir.exists():
        raise SystemExit(f"Embedding dir does not exist: {embedding_dir}")

    n_folds = int(train_args["n_folds"])
    seed = int(train_args["seed"])
    test_frac = float(train_args["test_frac"])

    test_ids, folds = cv_splits(embedding_dir, test_frac, n_folds, seed)
    if len(fold_dirs) != n_folds:
        print(f"Warning: found {len(fold_dirs)} fold dirs but args.n_folds={n_folds}. Using min().")
    n_use = min(len(fold_dirs), n_folds)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Scoring on device: {device}")

    metric_rows: list[dict] = []
    full_rows: list[dict] = []

    for k in range(n_use):
        fold_dir = args.runs_dir / f"fold_{k}"
        ckpt_path = fold_dir / "best.pt"
        if not ckpt_path.exists():
            print(f"Skipping missing checkpoint: {ckpt_path}")
            continue

        _, val_ids = folds[k]
        eval_ids = val_ids if args.split == "val" else test_ids
        result = evaluate_checkpoint_on_ids(ckpt_path, embedding_dir, eval_ids, device=device)

        for c in SS3_CHARS:
            metric_rows.append(
                {
                    "fold": k + 1,
                    "metric": "f1",
                    "class": c,
                    "value": float(result["prf1"][c]["f1"]),
                }
            )
            metric_rows.append(
                {
                    "fold": k + 1,
                    "metric": "sov",
                    "class": c,
                    "value": float(result["sov"]["per_class"][c]),
                }
            )

        full_rows.append(
            {
                "fold": k + 1,
                "split": args.split,
                "q3": result["q3"],
                "f1_H": float(result["prf1"]["H"]["f1"]),
                "f1_E": float(result["prf1"]["E"]["f1"]),
                "f1_C": float(result["prf1"]["C"]["f1"]),
                "sov_H": result["sov"]["per_class"]["H"],
                "sov_E": result["sov"]["per_class"]["E"],
                "sov_C": result["sov"]["per_class"]["C"],
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    full_df = pd.DataFrame(full_rows)
    metrics_df.to_csv(out_dir / "fold_metrics_long.csv", index=False)
    full_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    if not metrics_df.empty:
        plot_fold_metrics(metrics_df, out_dir)


if __name__ == "__main__":
    main()
