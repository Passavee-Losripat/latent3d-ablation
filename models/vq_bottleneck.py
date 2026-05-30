"""Shared vector-quantization bottleneck — representation-agnostic.

Quantizes a (B, 256, 4, 4, 4) continuous latent to the nearest entry in a
codebook of K vectors using EMA updates and dead-code restart.

EMA replaces the explicit codebook gradient loss: the embedding vectors track
exponential moving averages of the encoder outputs assigned to them. Dead-code
restart replaces unused vectors with random encoder outputs to prevent collapse.

This file must remain free of any representation-specific logic.

Tensor flow:
  Input  : (B, 256, 4, 4, 4)
  Reshape: (B*64, 256)          — 64 = 4*4*4 spatial positions
  Lookup : nearest codebook vector per position (L2)
  Output : (B, 256, 4, 4, 4)   quantized via straight-through estimator
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    z_q: torch.Tensor               # (B, 256, 4, 4, 4) quantized
    commitment_loss: torch.Tensor   # scalar — encoder output stays close to codebook
    codebook_loss: torch.Tensor     # scalar — always 0.0 (EMA handles codebook update)
    indices: torch.Tensor           # (B*64,) long
    usage_fraction: float           # fraction of codebook entries used in batch


class VQBottleneck(nn.Module):
    """EMA-updated VQ layer with dead-code restart."""

    def __init__(
        self,
        codebook_size: int = 512,
        embed_dim: int = 256,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        """
        Args:
            codebook_size: number of codebook vectors K.
            embed_dim:     dimension of each codebook vector D.
            decay:         EMA decay rate for cluster stats (0.99 = slow moving average).
            eps:           Laplace smoothing epsilon to prevent division by zero.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps

        # Initialize as L2-normalized random vectors for stable early training
        emb = F.normalize(torch.randn(codebook_size, embed_dim), dim=1)
        self.register_buffer("embedding",    emb)
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg",    emb.clone())

    def forward(self, z_e: torch.Tensor) -> VQOutput:
        """
        Args:
            z_e: (B, D, H, W, L) continuous encoder output.

        Returns:
            VQOutput with quantized tensor and associated losses.
        """
        B, D, H, W, L = z_e.shape
        # (B, D, H, W, L) → (B*H*W*L, D); cast to float for EMA stability
        z_flat = z_e.permute(0, 2, 3, 4, 1).contiguous().view(-1, D).float()

        # Squared L2 distances to all codebook entries
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (N, K)

        indices = distances.argmin(dim=1)  # (N,)
        z_q_flat = self.embedding[indices]  # (N, D)

        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.codebook_size).float()

                # EMA update of cluster sizes and embedding sums
                self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(one_hot.t() @ z_flat, alpha=1 - self.decay)

                # Laplace-smoothed cluster size for stable division
                n = self.cluster_size.sum()
                smoothed = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                self.embedding.copy_(self.embed_avg / smoothed.unsqueeze(1))

                # Dead-code restart: replace unused vectors with random encoder outputs
                dead = self.cluster_size < 0.5
                if dead.any():
                    n_dead = int(dead.sum().item())
                    rand_idx = torch.randint(0, z_flat.shape[0], (n_dead,), device=z_flat.device)
                    self.embedding[dead] = z_flat[rand_idx].detach()
                    self.cluster_size[dead] = 1.0
                    self.embed_avg[dead] = z_flat[rand_idx].detach()

        # Commitment loss: push encoder output toward the chosen codeword
        commitment_loss = F.mse_loss(z_q_flat.detach(), z_flat)
        # EMA handles the codebook update; no explicit codebook gradient loss needed
        codebook_loss = torch.tensor(0.0, device=z_flat.device)

        # Straight-through estimator: gradients flow through z_flat, not through indices
        z_q_flat_st = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat_st.view(B, H, W, L, D).permute(0, 4, 1, 2, 3).contiguous()

        # Cast back to original dtype (z_e may be fp16 under autocast)
        z_q = z_q.to(z_e.dtype)

        usage_fraction = indices.unique().numel() / self.codebook_size

        return VQOutput(
            z_q=z_q,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            indices=indices,
            usage_fraction=usage_fraction,
        )
