"""Shared 3D encoder — representation-agnostic.

Encodes a (B, 1, 32, 32, 32) voxel input into a (B, 256, 4, 4, 4) latent tensor
via four successive 3D convolutions with stride-2 downsampling.

This file must remain free of any representation-specific logic.

Layer shapes:
  Input   (B,   1, 32, 32, 32)
  Layer 0 (B,  32, 32, 32, 32)  — stride 1
  Layer 1 (B,  64, 16, 16, 16)  — stride 2
  Layer 2 (B, 128,  8,  8,  8)  — stride 2
  Layer 3 (B, 256,  4,  4,  4)  — stride 2
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Encoder(nn.Module):
    """Hierarchical 3D convolutional encoder with GroupNorm + SiLU activations."""

    def __init__(self, grad_checkpoint: bool = False) -> None:
        """
        Args:
            grad_checkpoint: if True, use gradient checkpointing on the
                             strided layers to save VRAM at the cost of
                             extra compute.
        """
        super().__init__()
        self.grad_checkpoint = grad_checkpoint

        self.layer0 = _ConvBlock(1,   32,  stride=1)
        self.layer1 = _ConvBlock(32,  64,  stride=2)
        self.layer2 = _ConvBlock(64,  128, stride=2)
        self.layer3 = _ConvBlock(128, 256, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 32, 32, 32) → (B, 256, 4, 4, 4)"""
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint(self.layer0, x, use_reentrant=False)
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
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
