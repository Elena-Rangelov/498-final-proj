"""
Step 4: Dataset that loads cached ESM embeddings and aligned SS3 labels.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

import importlib.util
_spec = importlib.util.spec_from_file_location("model", Path(__file__).parent / "03_model.py")
_model_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_model_mod)
SS3_TO_IDX = _model_mod.SS3_TO_IDX
PAD_IDX = _model_mod.PAD_IDX


class CachedEmbeddingDataset(Dataset):
    """
    Each item on disk is a dict:
        {"pdb_id": str, "sequence": str, "ss3": str, "embedding": fp16 tensor}
    """

    def __init__(self, embedding_dir: str | Path, pdb_ids: list[str] | None = None):
        self.dir = Path(embedding_dir)
        if pdb_ids is None:
            self.files = sorted(self.dir.glob("*.pt"))
        else:
            self.files = [self.dir / f"{pid}.pt" for pid in pdb_ids]
            self.files = [f for f in self.files if f.exists()]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        item = torch.load(self.files[idx], weights_only=False)
        emb = item["embedding"].float()  # (L, d), upcast from fp16
        labels = torch.tensor(
            [SS3_TO_IDX[c] for c in item["ss3"]], dtype=torch.long
        )
        return emb, labels


def pad_collate(batch):
    """
    Pad a batch of variable-length sequences.

    Returns:
        emb:    (B, L_max, d)         float
        labels: (B, L_max)            long  (PAD_IDX in padded positions)
        mask:   (B, L_max)            bool  (True for real residues)
    """
    embeddings, labels_list = zip(*batch)
    B = len(batch)
    L_max = max(e.shape[0] for e in embeddings)
    d = embeddings[0].shape[1]

    emb_padded = torch.zeros(B, L_max, d)
    label_padded = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, (e, l) in enumerate(zip(embeddings, labels_list)):
        L = e.shape[0]
        emb_padded[i, :L] = e
        label_padded[i, :L] = l
        mask[i, :L] = True

    return emb_padded, label_padded, mask
