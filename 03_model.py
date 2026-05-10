"""
Step 3: Define the model.

Architecture (mirrors the encoder side of "Attention Is All You Need"):

    ESM-2 embedding (frozen, dim=1280)
        |
    Linear projection: 1280 -> d_model
        |
    + Sinusoidal positional encoding
        |
    N x TransformerEncoderLayer
        - Multi-head self-attention (h heads)
        - Add & LayerNorm
        - Position-wise FFN (d_model -> d_ff -> d_model)
        - Add & LayerNorm
        |
    Linear: d_model -> 3
        |
    Logits over {H, E, C}

We use PyTorch's nn.TransformerEncoder, which implements exactly the encoder
block from Vaswani et al. 2017.
"""

import math
import torch
import torch.nn as nn


# Mapping from secondary structure character to class index
SS3_CHARS = ["H", "E", "C"]
SS3_TO_IDX = {c: i for i, c in enumerate(SS3_CHARS)}
IDX_TO_SS3 = {i: c for c, i in SS3_TO_IDX.items()}
NUM_CLASSES = 3
PAD_IDX = -100  # ignored by CrossEntropyLoss


class SinusoidalPositionalEncoding(nn.Module):
    """The fixed sinusoidal PE from Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model) so we can broadcast over batch.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class SSPredictor(nn.Module):
    """
    Per-residue 3-class secondary structure predictor.

    Input:  ESM embeddings of shape (B, L, esm_dim)
    Output: logits of shape (B, L, 3)
    """

    def __init__(
        self,
        esm_dim: int = 1280,        # 1280 for ESM-2 650M
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.input_proj = nn.Linear(esm_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # post-LN, matching the original paper
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Linear(d_model, NUM_CLASSES)

    def forward(
        self,
        esm_emb: torch.Tensor,            # (B, L, esm_dim)
        attention_mask: torch.Tensor,     # (B, L) bool, True for real residues
    ) -> torch.Tensor:
        x = self.input_proj(esm_emb)
        x = self.pos_enc(x)
        x = self.input_dropout(x)

        # nn.TransformerEncoder expects src_key_padding_mask where True = ignore
        key_padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        return self.classifier(x)  # (B, L, 3)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check
    model = SSPredictor()
    print(f"Trainable parameters: {count_params(model):,}")

    B, L, D = 2, 100, 1280
    emb = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    out = model(emb, mask)
    print(f"Input:  {emb.shape}")
    print(f"Output: {out.shape}")
