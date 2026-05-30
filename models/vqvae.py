"""VQ-VAE: assembles encoder + VQ bottleneck + latent projection + decoder.

The decoder is selected at runtime via the decoder registry, so this file
never needs to change when adding new representation variants.

Latent projection: a pair of 1×1×1 convolutions compress the 256-channel VQ
output to a compact latent_channels-channel tensor for diffusion training, then
expand back before the decoder. This keeps the VQ codebook large (256-D, which
is well-studied) while reducing the diffusion model's input dimensionality.

Forward tensor flow:
  Input    (B,   1, R, R, R)
  Encoder  (B, 256, 4, 4, 4)
  VQ       (B, 256, 4, 4, 4)  + vq losses
  proj     (B,  C,  4, 4, 4)  ← compact latent (C = latent_channels)
  unproj   (B, 256, 4, 4, 4)
  Decoder  (B,   1, R, R, R)
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
    reconstruction: torch.Tensor   # (B, 1, R, R, R) — R = voxel resolution
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    indices: torch.Tensor
    usage_fraction: float


class VQVAE(nn.Module):
    """Full VQ-VAE: encoder → VQ → proj → unproj → decoder."""

    def __init__(self, config: Any) -> None:
        """
        Args:
            config: namespace/dict-like with fields:
                      config.model.decoder          (str)
                      config.model.codebook_size    (int)
                      config.model.latent_channels  (int, default 64)
                      config.model.resolution       (int, default 64)
                      config.training.grad_checkpoint (bool)
        """
        super().__init__()
        model_cfg = config.model
        train_cfg = config.training

        resolution = getattr(model_cfg, "resolution", 64)
        latent_channels = getattr(model_cfg, "latent_channels", 64)
        grad_checkpoint = getattr(train_cfg, "grad_checkpoint", False)

        self.encoder = Encoder(resolution=resolution, grad_checkpoint=grad_checkpoint)
        self.vq = VQBottleneck(codebook_size=model_cfg.codebook_size, embed_dim=256)
        # 1×1×1 convolutions: no spatial mixing, just channel projection
        self.proj   = nn.Conv3d(256, latent_channels, kernel_size=1)
        self.unproj = nn.Conv3d(latent_channels, 256, kernel_size=1)
        self.decoder = get_decoder(model_cfg.decoder)

    def forward(self, x: torch.Tensor) -> VQVAEOutput:
        """x: (B, 1, R, R, R) → VQVAEOutput"""
        z_e = self.encoder(x)           # (B, 256, 4, 4, 4)
        vq_out: VQOutput = self.vq(z_e) # (B, 256, 4, 4, 4) quantized
        z_c = self.proj(vq_out.z_q)     # (B, C,   4, 4, 4) compact
        z_d = self.unproj(z_c)          # (B, 256, 4, 4, 4) expanded
        reconstruction = self.decoder(z_d)  # (B, 1, R, R, R)

        return VQVAEOutput(
            reconstruction=reconstruction,
            commitment_loss=vq_out.commitment_loss,
            codebook_loss=vq_out.codebook_loss,
            indices=vq_out.indices,
            usage_fraction=vq_out.usage_fraction,
        )

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + quantize + project; return (B, latent_channels, 4, 4, 4)."""
        z_e = self.encoder(x)
        return self.proj(self.vq(z_e).z_q)

    @torch.no_grad()
    def decode_latent(self, z_c: torch.Tensor) -> torch.Tensor:
        """Expand compact latent and decode: (B, latent_channels, 4, 4, 4) → (B, 1, R, R, R)."""
        return self.decoder(self.unproj(z_c))
