"""VQ-VAE: assembles encoder + VQ bottleneck + representation-specific decoder.

The decoder is selected at runtime via the decoder registry, so this file
never needs to change when adding new representation variants.

Forward tensor flow:
  Input  (B, 1, 32, 32, 32)
  → Encoder   → (B, 256, 4, 4, 4)
  → VQ        → (B, 256, 4, 4, 4)  + vq losses
  → Decoder   → (B, 1, 32, 32, 32)
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from models.encoder import Encoder
from models.vq_bottleneck import VQBottleneck, VQOutput
from models.decoders import get_decoder


@dataclass
class VQVAEOutput:
    reconstruction: torch.Tensor   # (B, 1, 32, 32, 32)
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    indices: torch.Tensor
    usage_fraction: float


class VQVAE(nn.Module):
    """Full VQ-VAE: encoder → VQ → decoder."""

    def __init__(self, config: Any) -> None:
        """
        Args:
            config: namespace/dict-like with fields:
                      config.model.decoder        (str)
                      config.model.codebook_size  (int)
                      config.training.grad_checkpoint (bool)
        """
        super().__init__()
        model_cfg = config.model
        train_cfg = config.training

        grad_checkpoint = getattr(train_cfg, "grad_checkpoint", False)

        self.encoder = Encoder(grad_checkpoint=grad_checkpoint)
        self.vq = VQBottleneck(
            codebook_size=model_cfg.codebook_size,
            embed_dim=256,
        )
        self.decoder = get_decoder(model_cfg.decoder)

    def forward(self, x: torch.Tensor) -> VQVAEOutput:
        """x: (B, 1, 32, 32, 32) → VQVAEOutput"""
        z_e = self.encoder(x)
        vq_out: VQOutput = self.vq(z_e)
        reconstruction = self.decoder(vq_out.z_q)

        return VQVAEOutput(
            reconstruction=reconstruction,
            commitment_loss=vq_out.commitment_loss,
            codebook_loss=vq_out.codebook_loss,
            indices=vq_out.indices,
            usage_fraction=vq_out.usage_fraction,
        )

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + quantize; return (B, 256, 4, 4, 4) for diffusion training."""
        z_e = self.encoder(x)
        return self.vq(z_e).z_q

    def decode_latent(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent (B, 256, 4, 4, 4) → (B, 1, 32, 32, 32)."""
        return self.decoder(z_q)
