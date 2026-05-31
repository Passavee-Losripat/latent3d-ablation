"""Triplane decoder: maps (B, 256, 4, 4, 4) latents to (B, 1, 64, 64, 64) TSDF volumes.

Instead of 3D transposed convolutions, this decoder projects the 4³ latent onto
three axis-aligned 2D feature planes (XY, XZ, YZ), upsamples each plane to 64×64
with 2D convolutions, then broadcasts all three back to 3D and sums them before
a small MLP head. This mirrors the EG3D/Triplane Diffusion architecture.

Memory grows O(N²) for the planes vs O(N³) for voxels — the key efficiency claim.

Tensor flow:
  Input          (B, 256,  4,  4,  4)
  Pool → 3 planes  each (B, 256, 4, 4) via max-pool over the missing axis
  2D upsample    each (B, 32, 64, 64)  via 4× stride-2 ConvTranspose2d
  Broadcast→3D   each (B, 32, 64, 64, 64) via expand (no copy)
  Sum            (B, 32, 64, 64, 64)
  MLP head       (B,  1, 64, 64, 64)  via 1×1×1 convs + tanh
"""

import torch
import torch.nn as nn


_PLANE_CHANNELS = 32   # feature channels per plane; 32 keeps peak VRAM manageable


class TriplaneDecoder(nn.Module):
    """Triplane-based decoder for TSDF reconstruction at 64³ resolution."""

    def __init__(self, plane_channels: int = _PLANE_CHANNELS) -> None:
        super().__init__()
        self.plane_channels = plane_channels

        # Three independent 2D upsampling networks: 4×4 → 64×64 (4 stride-2 steps)
        self.plane_xy = _Plane2DNet(256, plane_channels)   # pool over Z
        self.plane_xz = _Plane2DNet(256, plane_channels)   # pool over Y
        self.plane_yz = _Plane2DNet(256, plane_channels)   # pool over X

        # Lightweight MLP (1×1×1 convs — no spatial mixing, just channel projection)
        self.mlp = nn.Sequential(
            nn.Conv3d(plane_channels, 64, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Tanh(),   # TSDF output in [-1, 1]
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, 256, 4, 4, 4) → (B, 1, 64, 64, 64)"""
        B = z_q.shape[0]
        R = 64   # output resolution

        # Project 4³ latent onto three 2D planes via max-pooling over each axis
        # Convention: z_q has shape (B, C, Z, Y, X)
        feat_xy = z_q.max(dim=2).values   # (B, C, Y, X) — collapsed Z
        feat_xz = z_q.max(dim=3).values   # (B, C, Z, X) — collapsed Y
        feat_yz = z_q.max(dim=4).values   # (B, C, Z, Y) — collapsed X

        # Upsample each plane: (B, C, 4, 4) → (B, plane_ch, 64, 64)
        p_xy = self.plane_xy(feat_xy)   # (B, plane_ch, 64, 64)
        p_xz = self.plane_xz(feat_xz)
        p_yz = self.plane_yz(feat_yz)

        # Broadcast 2D planes → 3D volume (expand is zero-copy)
        # XY plane: same feature at every Z slice
        f_xy = p_xy.unsqueeze(2).expand(B, self.plane_channels, R, R, R)
        # XZ plane: same feature at every Y slice
        f_xz = p_xz.unsqueeze(3).expand(B, self.plane_channels, R, R, R)
        # YZ plane: same feature at every X slice
        f_yz = p_yz.unsqueeze(4).expand(B, self.plane_channels, R, R, R)

        # Sum triplane features and decode to scalar TSDF
        feat = f_xy + f_xz + f_yz   # (B, plane_ch, 64, 64, 64)
        return self.mlp(feat)        # (B, 1, 64, 64, 64)


class _Plane2DNet(nn.Module):
    """Four stride-2 2D transposed conv blocks: (B, 256, 4, 4) → (B, out_ch, 64, 64)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _UpBlock2D(in_ch,  128),   # 4  →  8
            _UpBlock2D(128,   64),    # 8  → 16
            _UpBlock2D(64,    out_ch),  # 16 → 32
            _UpBlock2D(out_ch, out_ch), # 32 → 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock2D(nn.Module):
    """ConvTranspose2d (stride=2, kernel=4) → GroupNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
