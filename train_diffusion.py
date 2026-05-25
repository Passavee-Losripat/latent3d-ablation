"""Stage 2 training: latent diffusion on VQ-VAE compressed representations.

Loads pre-saved latents from data/latents/{representation}/ (produced by
train_vqvae.py --extract-latents) and trains a 3D U-Net with DDPM objective.

At inference: DDIM sample → VQ-VAE decoder → marching cubes mesh.

Usage:
    python train_diffusion.py --config configs/tsdf.yaml [--vqvae_ckpt checkpoints/tsdf/best.pt]
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from diffusion.ddpm import DDPM
from diffusion.unet3d import UNet3D
from diffusion.ddim import DDIMSampler
from models.vqvae import VQVAE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _dict_to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        return _dict_to_ns(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Latent dataset
# ---------------------------------------------------------------------------

def load_latent_dataset(latent_dir: str | Path) -> TensorDataset:
    """Load all .npy latent files from a directory into a TensorDataset."""
    latent_dir = Path(latent_dir)
    files = sorted(latent_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No latent .npy files found in {latent_dir}.")

    arrays = [np.load(str(f)) for f in files]
    tensor = torch.from_numpy(np.stack(arrays, axis=0))  # (N, C, H, W, L)
    return TensorDataset(tensor)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str, vqvae_ckpt: str | None = None) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    diff_cfg = config.diffusion
    ckpt_dir = Path(diff_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs"))

    # Load latents
    ds = load_latent_dataset(diff_cfg.latent_dir)
    loader = DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    sample_latent = ds[0][0]  # (C, H, W, L)
    in_channels = sample_latent.shape[0]
    print(f"Loaded {len(ds)} latents of shape {tuple(sample_latent.shape)}")

    # Diffusion + UNet
    ddpm = DDPM(timesteps=diff_cfg.timesteps).to(device)
    unet = UNet3D(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=diff_cfg.lr)
    scaler = GradScaler(enabled=config.training.mixed_precision)

    global_step = 0
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        state = torch.load(str(latest_ckpt), map_location=device)
        unet.load_state_dict(state["unet"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        global_step = state["step"]
        print(f"Resumed from step {global_step}")

    t0 = time.time()
    data_iter = iter(loader)

    while global_step < diff_cfg.num_steps:
        try:
            (x0,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (x0,) = next(data_iter)

        x0 = x0.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config.training.mixed_precision):
            loss = ddpm.p_loss(unet, x0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        if global_step % 100 == 0:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0
            print(
                f"step {global_step:6d} | loss {loss.item():.6f} | "
                f"VRAM {vram_mb:.0f}MB | {elapsed:.0f}s"
            )
            writer.add_scalar("diffusion/loss", loss.item(), global_step)

        if global_step % diff_cfg.save_every == 0:
            state = {
                "unet": unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": global_step,
            }
            torch.save(state, str(ckpt_dir / "latest.pt"))
            torch.save(state, str(ckpt_dir / f"step_{global_step:06d}.pt"))
            print(f"  Saved checkpoint at step {global_step}")

    writer.close()
    print("Diffusion training complete.")

    # Quick inference demo (requires VQ-VAE decoder)
    if vqvae_ckpt:
        _inference_demo(config, ddpm, unet, device, vqvae_ckpt, diff_cfg)


def _inference_demo(
    config: SimpleNamespace,
    ddpm: DDPM,
    unet: UNet3D,
    device: torch.device,
    vqvae_ckpt: str,
    diff_cfg: SimpleNamespace,
) -> None:
    """Sample 4 latents, decode through VQ-VAE, save as .npy TSDF."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("scikit-image not installed; skipping mesh extraction demo.")
        return

    vqvae = VQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae.eval()

    sampler = DDIMSampler(ddpm, ddim_steps=diff_cfg.ddim_steps)

    sample_latent_path = next(Path(diff_cfg.latent_dir).glob("*.npy"), None)
    if sample_latent_path is None:
        return
    c, h, w, l = np.load(str(sample_latent_path)).shape
    shape = (4, c, h, w, l)

    print(f"Sampling {shape[0]} latents via DDIM ({diff_cfg.ddim_steps} steps)...")
    unet.eval()
    z_sampled = sampler.sample(unet, shape, device)

    out_dir = Path("outputs") / config.representation
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        tsdf_batch = vqvae.decode_latent(z_sampled)  # (B, 1, 32, 32, 32)

    for i in range(tsdf_batch.shape[0]):
        tsdf = tsdf_batch[i, 0].cpu().numpy()
        np.save(str(out_dir / f"generated_{i:04d}.npy"), tsdf.astype(np.float32))

    print(f"Saved generated TSDFs to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vqvae_ckpt", type=str, default=None)
    args = parser.parse_args()
    train(args.config, args.vqvae_ckpt)
