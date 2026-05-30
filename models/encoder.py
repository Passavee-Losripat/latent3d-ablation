"""Shared 3D encoder — representation-agnostic.

Encodes a voxel input into a (B, 256, 4, 4, 4) latent tensor via strided
3D convolutions. Depth scales with input resolution:

  resolution=32 → 4 blocks: 32 → 16 → 8 → 4  (stride-2 × 3)
  resolution=64 → 5 blocks: 64 → 32 → 16 → 8 → 4  (stride-2 × 4)

This file must remain free of any representation-specific logic.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Encoder(nn.Module):
    """Hierarchical 3D convolutional encoder with GroupNorm + SiLU activations."""

    def __init__(self, resolution: int = 64, grad_checkpoint: bool = False) -> None:
        """
        Args:
            resolution:       input voxel resolution (32 or 64).
            grad_checkpoint:  if True, use gradient checkpointing on strided layers
                              to save VRAM at the cost of extra compute.
        """
        super().__init__()
        self.grad_checkpoint = grad_checkpoint

        self.layer0 = _ConvBlock(1,   32,  stride=1)   # (B,  32, R,   R,   R  )
        self.layer1 = _ConvBlock(32,  64,  stride=2)   # (B,  64, R/2, R/2, R/2)
        self.layer2 = _ConvBlock(64,  128, stride=2)   # (B, 128, R/4, R/4, R/4)
        self.layer3 = _ConvBlock(128, 256, stride=2)   # (B, 256, R/8, R/8, R/8)
        # Extra stride-2 block only needed for 64³ to reach the 4³ bottleneck
        self.layer4 = _ConvBlock(256, 256, stride=2) if resolution == 64 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, R, R, R) → (B, 256, 4, 4, 4)"""
        layers = [self.layer0, self.layer1, self.layer2, self.layer3]
        if self.layer4 is not None:
            layers.append(self.layer4)

        if self.grad_checkpoint and x.requires_grad:
            for layer in layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in layers:
                x = layer(x)
        return x


class _ConvBlock(nn.Module):
    """Conv3d → GroupNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
