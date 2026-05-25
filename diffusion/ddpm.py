"""Denoising Diffusion Probabilistic Model (DDPM) with linear noise schedule.

Operates on latent tensors of shape (B, C, H, W, L) (e.g. (B, 256, 4, 4, 4)).

Noise schedule: linear beta schedule from beta_start to beta_end over T=1000 steps.
Parameterization: epsilon (predict the noise added at each timestep).
Training loss: MSE between predicted and actual noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    """DDPM forward process and training loss; inference handled by DDIMSampler."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps

        # Pre-compute schedule buffers (not trained)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample x_t from x_0 at timestep t.

        Returns:
            x_t:   noisy latent
            noise: the Gaussian noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Expand schedule coefficients to broadcast over (B, C, H, W, L)
        def _coef(buf: torch.Tensor) -> torch.Tensor:
            return buf[t].view(-1, *([1] * (x0.ndim - 1)))

        x_t = _coef(self.sqrt_alphas_cumprod) * x0 + _coef(self.sqrt_one_minus_alphas_cumprod) * noise
        return x_t, noise

    def p_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Sample random timesteps, add noise, compute epsilon-prediction MSE loss."""
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)
        noise_pred = model(x_t, t)
        return F.mse_loss(noise_pred, noise)
