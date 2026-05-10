"""
Step 6: Predict secondary structure for a new sequence end-to-end:
        sequence -> ESM-2 embedding -> SSPredictor -> H/E/C string.
"""

import argparse
import importlib.util
from pathlib import Path

import torch
import esm


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


_model_mod = _load("model", "03_model.py")
SSPredictor = _model_mod.SSPredictor
IDX_TO_SS3 = _model_mod.IDX_TO_SS3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt from training.")
    ap.add_argument("--sequence", required=True, help="Amino acid sequence (one-letter).")
    ap.add_argument("--esm_model", default="esm2_t33_650M_UR50D",
                    help="MUST match the model used to create training embeddings.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ESM
    print(f"Loading {args.esm_model}...")
    model_loader = getattr(esm.pretrained, args.esm_model)
    esm_model, alphabet = model_loader()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval().to(device)

    # Embed the input sequence
    seq = args.sequence.strip().upper()
    _, _, tokens = batch_converter([("query", seq)])
    tokens = tokens.to(device)
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[esm_model.num_layers])
    emb = out["representations"][esm_model.num_layers][0, 1 : len(seq) + 1]  # (L, d)

    # Load head
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head_args = ckpt["args"]
    head = SSPredictor(
        esm_dim=head_args["esm_dim"], d_model=head_args["d_model"],
        n_heads=head_args["n_heads"], n_layers=head_args["n_layers"],
        d_ff=head_args["d_ff"], dropout=0.0,
    ).to(device)
    head.load_state_dict(ckpt["model_state"])
    head.eval()

    # Predict
    with torch.no_grad():
        emb_b = emb.unsqueeze(0)
        mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
        logits = head(emb_b, mask)
        preds = logits.argmax(dim=-1)[0].cpu().tolist()

    ss3 = "".join(IDX_TO_SS3[i] for i in preds)
    print(f"\nSeq: {seq}")
    print(f"SS:  {ss3}")


if __name__ == "__main__":
    main()
