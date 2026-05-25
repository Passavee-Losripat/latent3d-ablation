"""Shared vector-quantization bottleneck — representation-agnostic.

Quantizes a (B, 256, 4, 4, 4) continuous latent to the nearest entry in a
learned codebook of K=512 vectors, using a straight-through gradient estimator.

This file must remain free of any representation-specific logic.

Tensor flow:
  Input  : (B, 256, 4, 4, 4)
  Reshape: (B*64, 256)          — 64 = 4*4*4 spatial positions
  Lookup : nearest codebook vector per position
  Output : (B, 256, 4, 4, 4)   quantized
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    z_q: torch.Tensor          # (B, 256, 4, 4, 4) quantized
    commitment_loss: torch.Tensor  # scalar
    codebook_loss: torch.Tensor    # scalar
    indices: torch.Tensor      # (B*64,) long
    usage_fraction: float      # fraction of codebook entries used in batch


class VQBottleneck(nn.Module):
    """EMA-free VQ layer with straight-through estimator."""

    def __init__(self, codebook_size: int = 512, embed_dim: int = 256) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim

        # Codebook: K vectors of dimension D
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor) -> VQOutput:
        """
        Args:
            z_e: (B, D, H, W, L) continuous encoder output.

        Returns:
            VQOutput with quantized tensor and associated losses.
        """
        B, D, H, W, L = z_e.shape
        # (B, D, H, W, L) → (B*H*W*L, D)
        z_flat = z_e.permute(0, 2, 3, 4, 1).contiguous().view(-1, D)

        # Squared L2 distances to all codebook entries
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )  # (N, K)

        indices = distances.argmin(dim=1)  # (N,)
        z_q_flat = self.embedding(indices)  # (N, D)

        # Losses (straight-through: gradients flow through z_e, not indices)
        commitment_loss = F.mse_loss(z_q_flat.detach(), z_flat)
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())

        # Straight-through estimator
        z_q_flat_st = z_flat + (z_q_flat - z_flat).detach()

        # Reshape back to spatial
        z_q = z_q_flat_st.view(B, H, W, L, D).permute(0, 4, 1, 2, 3).contiguous()

        usage_fraction = indices.unique().numel() / self.codebook_size

        return VQOutput(
            z_q=z_q,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            indices=indices,
            usage_fraction=usage_fraction,
        )
