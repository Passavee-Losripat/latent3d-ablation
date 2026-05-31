"""Occupancy decoder: maps (B, 256, 4, 4, 4) latents to (B, 1, 64, 64, 64) occupancy grids.

Output values are in [0.0, 1.0] via sigmoid, representing inside probability.
Loss: binary cross-entropy against binary voxel grids (data/processed/voxel_64/).

Layer shapes:
  Input   (B, 256,  4,  4,  4)
  Block 0 (B, 256,  8,  8,  8)  — ConvTranspose3d stride 2
  Block 1 (B, 128, 16, 16, 16)  — ConvTranspose3d stride 2
  Block 2 (B,  64, 32, 32, 32)  — ConvTranspose3d stride 2
  Block 3 (B,  32, 64, 64, 64)  — ConvTranspose3d stride 2
  Head    (B,   1, 64, 64, 64)  — Conv3d + sigmoid
"""

import torch
import torch.nn as nn


class OccupancyDecoder(nn.Module):
    """3D transposed-conv decoder for binary occupancy reconstruction at 64³."""

    def __init__(self) -> None:
        super().__init__()

        self.block0 = _UpBlock(256, 256)   #  4³ →  8³
        self.block1 = _UpBlock(256, 128)   #  8³ → 16³
        self.block2 = _UpBlock(128, 64)    # 16³ → 32³
        self.block3 = _UpBlock(64,  32)    # 32³ → 64³

        self.head = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),   # output in [0, 1] — inside probability
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, 256, 4, 4, 4) → (B, 1, 64, 64, 64)"""
        x = self.block0(z_q)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


class _UpBlock(nn.Module):
    """ConvTranspose3d (stride=2, kernel=4) → GroupNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
