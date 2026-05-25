"""Lightweight 3D U-Net for diffusion on (B, C, 4, 4, 4) latents.

Designed to fit inside 12GB VRAM alongside the VQ-VAE decoder.

Architecture:
  - Sinusoidal + MLP time embedding injected at every block via scale-shift
  - 2 encoder blocks (down), 1 bottleneck, 2 decoder blocks (up) with skip connections
  - All spatial ops are 3D conv; no attention (keeps memory low for 4³ feature maps)

Tensor flow:
  Input  (B, in_channels, 4, 4, 4)  with t: (B,)
  Down0  (B, base_ch,     4, 4, 4)   — no spatial downsampling (latent is already 4³)
  Down1  (B, base_ch*2,   4, 4, 4)
  Bot    (B, base_ch*4,   4, 4, 4)
  Up1    (B, base_ch*2,   4, 4, 4)   + skip from Down1
  Up0    (B, base_ch,     4, 4, 4)   + skip from Down0
  Output (B, in_channels, 4, 4, 4)
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, dim) float"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


class TimeEmbMLP(nn.Module):
    """Project sinusoidal embedding to 4×dim for scale-shift conditioning."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)  # (B, 4*dim)


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """Residual block with time-conditional scale-shift (AdaGN)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.act = nn.SiLU()

        # Projects time embedding to scale + shift for norm2
        self.time_proj = nn.Linear(time_dim, out_ch * 2)

        self.skip = (
            nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Scale-shift from time embedding
        scale_shift = self.time_proj(self.act(time_emb))  # (B, 2*out_ch)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """Lightweight 3D U-Net for latent diffusion on 4³ feature maps."""

    def __init__(
        self,
        in_channels: int = 256,
        base_channels: int = 128,
    ) -> None:
        super().__init__()
        ch = base_channels
        time_dim = ch * 4  # dimension of the projected time embedding

        self.time_mlp = TimeEmbMLP(ch)

        # Encoder (no spatial downsampling — latent is already 4³)
        self.down0 = ResBlock3D(in_channels, ch,     time_dim)
        self.down1 = ResBlock3D(ch,           ch * 2, time_dim)

        # Bottleneck
        self.bot0  = ResBlock3D(ch * 2, ch * 4, time_dim)
        self.bot1  = ResBlock3D(ch * 4, ch * 2, time_dim)

        # Decoder with skip connections
        self.up1 = ResBlock3D(ch * 2 + ch * 2, ch * 2, time_dim)
        self.up0 = ResBlock3D(ch * 2 + ch,     ch,     time_dim)

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv3d(ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, 4, 4, 4) noisy latent
            t: (B,) long timestep indices

        Returns:
            (B, in_channels, 4, 4, 4) predicted noise
        """
        te = self.time_mlp(t)  # (B, time_dim)

        h0 = self.down0(x,  te)  # (B, ch,   4, 4, 4)
        h1 = self.down1(h0, te)  # (B, ch*2, 4, 4, 4)

        h = self.bot0(h1, te)    # (B, ch*4, 4, 4, 4)
        h = self.bot1(h,  te)    # (B, ch*2, 4, 4, 4)

        h = self.up1(torch.cat([h, h1], dim=1), te)  # (B, ch*2, 4, 4, 4)
        h = self.up0(torch.cat([h, h0], dim=1), te)  # (B, ch,   4, 4, 4)

        return self.out_conv(nn.functional.silu(self.out_norm(h)))
