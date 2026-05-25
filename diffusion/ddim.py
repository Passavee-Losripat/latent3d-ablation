"""DDIM deterministic sampler for latent diffusion.

Works with any DDPM-trained epsilon-prediction network without retraining.
Uses 50 inference steps (configurable) via uniform timestep skipping.

Reference: Song et al., "Denoising Diffusion Implicit Models" (2020).

Input/output shapes mirror the DDPM latent space: (B, C, H, W, L).
"""

import torch
import torch.nn as nn


class DDIMSampler:
    """Deterministic DDIM sampler wrapping a trained DDPM schedule."""

    def __init__(self, ddpm: nn.Module, ddim_steps: int = 50) -> None:
        """
        Args:
            ddpm:       trained DDPM instance (provides alphas_cumprod buffer).
            ddim_steps: number of denoising steps at inference time.
        """
        self.ddpm = ddpm
        self.ddim_steps = ddim_steps

        T = ddpm.timesteps
        # Uniformly spaced timestep indices in descending order
        step_size = T // ddim_steps
        self.timesteps = list(range(T - 1, -1, -step_size))[:ddim_steps]

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate samples by iterative DDIM denoising.

        Args:
            model:  3D U-Net (takes x_t and t, returns predicted noise).
            shape:  output shape, e.g. (B, 256, 4, 4, 4).
            device: torch device.
            eta:    stochasticity factor; 0.0 = fully deterministic DDIM.

        Returns:
            x0 estimate of shape `shape`.
        """
        alphas = self.ddpm.alphas_cumprod  # (T,)

        x = torch.randn(shape, device=device)

        for i, t_curr in enumerate(self.timesteps):
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1

            alpha_t = alphas[t_curr]
            alpha_prev = alphas[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            t_batch = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)
            eps = model(x, t_batch)

            # Predicted x0
            x0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM update
            sigma = (
                eta
                * ((1 - alpha_prev) / (1 - alpha_t)).sqrt()
                * (1 - alpha_t / alpha_prev).sqrt()
            )
            dir_xt = (1 - alpha_prev - sigma ** 2).sqrt() * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0

            x = alpha_prev.sqrt() * x0_pred + dir_xt + noise

        return x
