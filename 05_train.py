"""
Step 5: Train the SSPredictor head on cached ESM embeddings, evaluate on a
        held-out test split.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_dir", default="data/embeddings")
    ap.add_argument("--out_dir", default="runs/exp1")

    # Architecture
    ap.add_argument("--esm_dim", type=int, default=1280)  # 650M model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Splits
    train_ids, val_ids, test_ids = split_ids(
        Path(args.embedding_dir), args.val_frac, args.test_frac, args.seed
    )
    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    train_ds = CachedEmbeddingDataset(args.embedding_dir, train_ids)
    val_ds = CachedEmbeddingDataset(args.embedding_dir, val_ids)
    test_ds = CachedEmbeddingDataset(args.embedding_dir, test_ids)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=pad_collate,
    )

    # Model
    model = SSPredictor(
        esm_dim=args.esm_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    print(f"Trainable params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_q3 = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for emb, labels, mask in train_loader:
            emb, labels, mask = emb.to(device), labels.to(device), mask.to(device)
            logits = model(emb, mask)            # (B, L, 3)
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
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_q3={val_metrics['q3']:.4f} | "
            f"H={val_metrics['per_class']['H']['recall']:.3f} "
            f"E={val_metrics['per_class']['E']['recall']:.3f} "
            f"C={val_metrics['per_class']['C']['recall']:.3f}"
        )

        if val_metrics["q3"] > best_val_q3:
            best_val_q3 = val_metrics["q3"]
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args), "epoch": epoch},
                best_path,
            )

    # Final test evaluation
    print(f"\nBest val Q3: {best_val_q3:.4f}. Loading best checkpoint for test eval...")
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Q3 = {test_metrics['q3']:.4f}")
    for c in SS3_CHARS:
        pc = test_metrics["per_class"][c]
        print(f"  {c}: precision={pc['precision']:.3f}  recall={pc['recall']:.3f}")
    print(f"  Confusion (rows=true, cols=pred): {test_metrics['confusion']}")


if __name__ == "__main__":
    main()
