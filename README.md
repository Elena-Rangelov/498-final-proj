
# el notes
 - check the setup_and_troubleshooting doc to run
 - model rationale doc and image.png include the structure of the model and why
 - el_output.txt shows my output with the smaller model. it's honestly not bad and does a good job at continuing to climb without overfitting. bigger model would be better + could tweak the hyperparameters



# Protein Secondary Structure Prediction

Per-residue 3-class secondary structure prediction (H = α-helix, E = β-strand,
C = coil) using frozen ESM-2 embeddings followed by a small Transformer encoder
head.

## Pipeline

```
PDB IDs --> [01] download + DSSP -->  labels.tsv  (sequence, ss3 per protein)
                                          |
                                          v
                                 [02] frozen ESM-2 650M
                                          |
                                          v
                              data/embeddings/*.pt  (per-residue 1280-d vectors)
                                          |
                                          v
                                  [05] train head:
                                  Linear(1280 -> 256)
                                  + sinusoidal PE
                                  + 2x TransformerEncoderLayer (8 heads, d_ff=1024)
                                  + Linear(256 -> 3)
                                          |
                                          v
                              [06] predict H/E/C per residue
```

## Files

- `01_prepare_data.py` — download PDBs, run DSSP, write `labels.tsv`.
- `02_embed_with_esm.py` — pre-compute ESM-2 embeddings, cache to `data/embeddings/`.
- `03_model.py` — `SSPredictor` (Transformer encoder head, mirrors Vaswani et al. 2017).
- `04_dataset.py` — `Dataset` + padded collate over cached embeddings.
- `05_train.py` — train + validate + test on a random split.
- `06_predict.py` — end-to-end inference for a new sequence.

## Setup

```bash
pip install torch fair-esm biopython numpy
# DSSP binary:
sudo apt-get install dssp           # or: conda install -c salilab dssp
```

## Quick start

```bash
# 1. Download PDBs and label them with DSSP
echo -e "1ubq\n2lyz\n1crn\n..." > pdb_list.txt   # 100+ IDs
python 01_prepare_data.py --pdb_list pdb_list.txt

# 2. Cache ESM-2 embeddings (one-time, ~few minutes on a GPU)
python 02_embed_with_esm.py

# 3. Train (fast: head trains in seconds/epoch on cached features)
python 05_train.py --epochs 30

# 4. Predict on a new sequence
python 06_predict.py --checkpoint runs/exp1/best.pt \
    --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK
```

## Why this architecture?

- **Frozen ESM-2 650M** for the residue embeddings. The hard work
  (capturing context, evolutionary information) is already done by the protein
  language model. Pre-trained on ~65M sequences via masked language modeling.
- **Cache embeddings to disk.** ESM is the expensive part; running it once
  decouples it from the training loop and makes head iteration nearly free.
- **Transformer encoder head**, mirroring "Attention Is All You Need":
  multi-head self-attention + position-wise FFN + residual + LayerNorm.
  Stacked 2 deep with d_model=256 — a few million parameters, easily fits
  in 16 GB VRAM, trains in minutes.
- **Sinusoidal positional encodings** added on top of ESM features. Strictly
  speaking ESM-2 already encodes position via rotary embeddings internally,
  but adding sinusoidal PE here makes the head architecturally identical to
  the paper.

## Notes on labels

DSSP returns 8 states; we collapse them with the standard convention:
- H, G, I -> H
- E, B    -> E
- everything else -> C

Sequences longer than 1022 residues are truncated to fit ESM-2's input limit.

## Expected results

With ~100-200 PDB structures and ESM-2 650M embeddings, a 2-layer
Transformer head typically gets Q3 in the low-to-mid 80s. A linear/MLP probe
on the same features is a strong baseline (also low-80s); the gap to the
encoder head is small because ESM has already done most of the contextual
reasoning. This is a good thing to discuss in the writeup.
