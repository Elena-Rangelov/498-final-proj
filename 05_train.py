"""
Step 5: Train the SSPredictor head on cached ESM embeddings.

Two execution paths (select with --no_cv / --cv):

  Default – 5-fold cross-validation
  ───────────────────────────────────
  1. Reserve a fixed held-out test set (--test_frac, default 15 %).
  2. Run KFold(n_splits=--n_folds) on the remaining data; each fold:
       • fresh model + optimizer + scheduler
       • 30 epochs, best-val checkpoint saved to <out_dir>/fold_k/best.pt
  3. Report per-fold and aggregate (mean ± std) Q3 on each fold's val set.
  4. Evaluate all fold checkpoints on the shared test set and report those too.

  Legacy – fixed train/val/test split (--no_cv)
  ───────────────────────────────────────────────
  Single train/val split; best checkpoint evaluated on held-out test set.
  Behaviour is identical to the original implementation.

Metrics:
  - Q3 accuracy: per-residue accuracy across {H, E, C} (the standard metric)
  - Per-class precision/recall
"""

import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_model_mod = _load("model", "03_model.py")
_data_mod = _load("dataset", "04_dataset.py")
SSPredictor = _model_mod.SSPredictor
count_params = _model_mod.count_params
PAD_IDX = _model_mod.PAD_IDX
SS3_CHARS = _model_mod.SS3_CHARS
CachedEmbeddingDataset = _data_mod.CachedEmbeddingDataset
pad_collate = _data_mod.pad_collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_ids(embedding_dir: Path, val_frac: float, test_frac: float, seed: int):
    ids = sorted(p.stem for p in embedding_dir.glob("*.pt"))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return train_ids, val_ids, test_ids


def cv_splits(
    embedding_dir: Path,
    test_frac: float,
    n_folds: int,
    seed: int,
) -> tuple[list[str], list[tuple[list[str], list[str]]]]:
    """
    Reserve a fixed test set, then yield KFold splits of the remainder.

    Returns
    -------
    test_ids  : list[str]  — held out for final evaluation only
    folds     : list of (train_ids, val_ids) tuples, one per fold
    """
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


def evaluate(model, loader, device) -> dict:
    """Return dict with q3 and per-class precision/recall."""
    model.eval()
    n_total, n_correct = 0, 0
    confusion = torch.zeros(3, 3, dtype=torch.long)

    with torch.no_grad():
        for emb, labels, mask in loader:
            emb, labels, mask = emb.to(device), labels.to(device), mask.to(device)
            logits = model(emb, mask)
            preds = logits.argmax(dim=-1)

            valid = mask
            preds = preds[valid]
            labels = labels[valid]

            n_correct += (preds == labels).sum().item()
            n_total += labels.numel()

            for t in range(3):
                for p in range(3):
                    confusion[t, p] += ((labels == t) & (preds == p)).sum().item()

    q3 = n_correct / max(n_total, 1)
    per_class = {}
    for c, name in enumerate(SS3_CHARS):
        tp = confusion[c, c].item()
        fn = confusion[c, :].sum().item() - tp
        fp = confusion[:, c].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[name] = {"precision": precision, "recall": recall}

    return {"q3": q3, "per_class": per_class, "confusion": confusion.tolist()}


def build_model(args, device) -> nn.Module:
    return SSPredictor(
        esm_dim=args.esm_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)


def make_loader(embedding_dir, ids, batch_size, shuffle, num_workers):
    ds = CachedEmbeddingDataset(embedding_dir, ids)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=pad_collate,
    )


def train_one_run(
    args,
    device: str,
    train_ids: list[str],
    val_ids: list[str],
    out_dir: Path,
    fold_label: str = "",
) -> tuple[float, dict]:
    """
    Train for args.epochs, checkpoint on val Q3.

    Returns (best_val_q3, val_metrics_at_best).
    Saves best checkpoint to out_dir/best.pt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        args.embedding_dir, train_ids, args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        args.embedding_dir, val_ids, args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_q3 = -1.0
    best_metrics: dict = {}
    best_path = out_dir / "best.pt"
    prefix = f"[{fold_label}] " if fold_label else ""

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0

        for emb, labels, mask in train_loader:
            emb, labels, mask = emb.to(device), labels.to(device), mask.to(device)
            logits = model(emb, mask)
            loss = criterion(logits.reshape(-1, 3), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"{prefix}Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_q3={val_metrics['q3']:.4f} | "
            f"H={val_metrics['per_class']['H']['recall']:.3f} "
            f"E={val_metrics['per_class']['E']['recall']:.3f} "
            f"C={val_metrics['per_class']['C']['recall']:.3f}"
        )

        if val_metrics["q3"] > best_val_q3:
            best_val_q3 = val_metrics["q3"]
            best_metrics = val_metrics
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args), "epoch": epoch},
                best_path,
            )

    return best_val_q3, best_metrics


def run_cv(args, device: str, out_dir: Path) -> None:
    """5-fold (or --n_folds) cross-validation with a shared held-out test set."""
    test_ids, folds = cv_splits(
        Path(args.embedding_dir), args.test_frac, args.n_folds, args.seed
    )
    print(
        f"CV setup: {args.n_folds} folds | "
        f"test={len(test_ids)} | "
        f"cv_pool={sum(len(t) + len(v) for t, v in folds[:1])} per fold total"
    )

    test_loader = make_loader(
        args.embedding_dir, test_ids, args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    fold_val_q3s: list[float] = []
    fold_test_q3s: list[float] = []
    fold_per_class: list[dict] = []

    for k, (train_ids, val_ids) in enumerate(folds):
        fold_dir = out_dir / f"fold_{k}"
        print(
            f"\n{'='*60}\n"
            f"  Fold {k+1}/{args.n_folds}  "
            f"train={len(train_ids)}  val={len(val_ids)}\n"
            f"{'='*60}"
        )
        set_seed(args.seed + k)  # different seed per fold, reproducible

        best_val_q3, _ = train_one_run(
            args, device, train_ids, val_ids, fold_dir,
            fold_label=f"fold {k+1}/{args.n_folds}",
        )
        fold_val_q3s.append(best_val_q3)

        # Evaluate this fold's best checkpoint on the shared test set
        ckpt = torch.load(fold_dir / "best.pt", weights_only=False)
        model = build_model(args, device)
        model.load_state_dict(ckpt["model_state"])
        test_metrics = evaluate(model, test_loader, device)
        fold_test_q3s.append(test_metrics["q3"])
        fold_per_class.append(test_metrics["per_class"])

        print(
            f"\n  → Fold {k+1} best val Q3 = {best_val_q3:.4f}  |  "
            f"test Q3 = {test_metrics['q3']:.4f}"
        )

    # Aggregate results
    val_arr  = np.array(fold_val_q3s)
    test_arr = np.array(fold_test_q3s)
    print(f"\n{'='*60}")
    print(f"Cross-validation summary ({args.n_folds} folds)")
    print(f"{'='*60}")
    print(f"  Val  Q3 : {val_arr.mean():.4f} ± {val_arr.std():.4f}  "
          f"(per fold: {', '.join(f'{v:.4f}' for v in fold_val_q3s)})")
    print(f"  Test Q3 : {test_arr.mean():.4f} ± {test_arr.std():.4f}  "
          f"(per fold: {', '.join(f'{v:.4f}' for v in fold_test_q3s)})")
    print()

    # Per-class test metrics averaged across folds
    for c in SS3_CHARS:
        prec = np.mean([f[c]["precision"] for f in fold_per_class])
        rec  = np.mean([f[c]["recall"]    for f in fold_per_class])
        print(f"  {c}: avg test precision={prec:.3f}  avg test recall={rec:.3f}")


def run_fixed_split(args, device: str, out_dir: Path) -> None:
    """Original single train/val/test split path."""
    train_ids, val_ids, test_ids = split_ids(
        Path(args.embedding_dir), args.val_frac, args.test_frac, args.seed
    )
    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    test_loader = make_loader(
        args.embedding_dir, test_ids, args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    best_val_q3, _ = train_one_run(
        args, device, train_ids, val_ids, out_dir,
    )

    print(f"\nBest val Q3: {best_val_q3:.4f}. Loading best checkpoint for test eval...")
    ckpt = torch.load(out_dir / "best.pt", weights_only=False)
    model = build_model(args, device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)

    print(f"\nTest Q3 = {test_metrics['q3']:.4f}")
    for c in SS3_CHARS:
        pc = test_metrics["per_class"][c]
        print(f"  {c}: precision={pc['precision']:.3f}  recall={pc['recall']:.3f}")
    print(f"  Confusion (rows=true, cols=pred): {test_metrics['confusion']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_dir", default="data/embeddings")
    ap.add_argument("--out_dir", default="runs/exp1")

    # Architecture
    ap.add_argument("--esm_dim",    type=int,   default=480)  # 480 for ESM-2 150M; 1280 for 650M
    ap.add_argument("--d_model",    type=int,   default=256)
    ap.add_argument("--n_heads",    type=int,   default=8)
    ap.add_argument("--n_layers",   type=int,   default=2)
    ap.add_argument("--d_ff",       type=int,   default=1024)
    ap.add_argument("--dropout",    type=float, default=0.1)

    # Training
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--test_frac",    type=float, default=0.15)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--num_workers",  type=int,   default=0)

    # CV vs fixed-split
    ap.add_argument(
        "--no_cv", dest="cv", action="store_false",
        help="Use fixed train/val/test split instead of cross-validation.",
    )
    ap.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of CV folds (ignored when --no_cv is set).",
    )
    # val_frac is only used by the fixed-split path
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.set_defaults(cv=True)

    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mode: {'5-fold cross-validation' if args.cv else 'fixed split'}")

    model_tmp = build_model(args, device)
    print(f"Trainable params: {count_params(model_tmp):,}")
    del model_tmp

    if args.cv:
        run_cv(args, device, out_dir)
    else:
        run_fixed_split(args, device, out_dir)


if __name__ == "__main__":
    main()
