"""
Step 2: Pre-compute ESM-2 embeddings for every sequence in labels.tsv
        and save them to disk.

Why cache?
  ESM-2 650M is the expensive part. We freeze it and run it ONCE over the
  training set. Then training the small head is fast (seconds per epoch),
  which lets us iterate on architecture freely.

Requires:
  pip install torch fair-esm
"""

# IMPORTANT: This must be set BEFORE importing torch.
# Conda on Windows often ships multiple OpenMP runtimes (one in numpy/MKL,
# one in torch). They conflict at import time. This env var tells OpenMP
# to allow duplicate runtimes — safe for transformer inference.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
from pathlib import Path

import torch
import esm  # fair-esm package


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_tsv", default="data/labels.tsv")
    ap.add_argument("--out_dir", default="data/embeddings")
    ap.add_argument("--model_name", default="esm2_t33_650M_UR50D",
                    help="ESM-2 variant. Use esm2_t12_35M_UR50D for a quick "
                         "sanity check, esm2_t33_650M_UR50D for real runs.")
    ap.add_argument("--max_len", type=int, default=1022,
                    help="Truncate sequences longer than this (ESM has a "
                         "limit; 1022 leaves room for BOS/EOS tokens).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_name}...")
    model_loader = getattr(esm.pretrained, args.model_name)
    model, alphabet = model_loader()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(args.device)

    # The last layer index depends on the model.
    # For esm2_t33_650M_UR50D, the last layer is 33.
    last_layer = model.num_layers
    print(f"Using representations from layer {last_layer}.")

    # Read all (id, seq, ss3) rows. Use "pdb_chain" as the unique key so that
    # different chains of the same PDB don't collide on disk.
    rows = []
    with open(args.labels_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            chain = r.get("chain", "") or ""
            uid = f"{r['pdb_id']}_{chain}" if chain else r["pdb_id"]
            rows.append((uid, r["sequence"], r["ss3"]))
    print(f"Embedding {len(rows)} sequences...")

    # Process one at a time to keep memory predictable.
    # ESM-2 650M on a single sequence of length ~1000 fits easily in 16GB.
    with torch.no_grad():
        for i, (uid, seq, ss3) in enumerate(rows, 1):
            # Truncate if needed (trim BOTH seq and label so they stay aligned)
            if len(seq) > args.max_len:
                seq = seq[: args.max_len]
                ss3 = ss3[: args.max_len]

            data = [(uid, seq)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(args.device)

            out = model(tokens, repr_layers=[last_layer])
            # Shape: (1, L+2, d).  Strip BOS (index 0) and EOS (index L+1).
            reps = out["representations"][last_layer][0, 1 : len(seq) + 1].cpu()

            assert reps.shape[0] == len(seq) == len(ss3), (
                f"length mismatch on {uid}: emb={reps.shape[0]}, "
                f"seq={len(seq)}, ss3={len(ss3)}"
            )

            torch.save(
                {"uid": uid, "sequence": seq, "ss3": ss3,
                 "embedding": reps.half()},  # save as fp16 to halve disk usage
                out_dir / f"{uid}.pt",
            )

            if i % 20 == 0:
                print(f"  [{i}/{len(rows)}] cached {uid} (L={len(seq)})")

    print(f"\nDone. Embeddings saved to {out_dir}")


if __name__ == "__main__":
    main()
