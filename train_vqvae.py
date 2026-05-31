"""Stage 1 training: VQ-VAE on 3D voxel representations.

Usage:
    python train_vqvae.py --config configs/tsdf.yaml

After training, saves compact latents for every shape to
data/latents/{representation}/<shape_id>.npy and writes latent_stats.npz
(mean + std) needed by the diffusion training stage.

Total loss:  L = L_recon_weighted + commitment_weight * L_commit
  (codebook gradient is handled by EMA — vq_weight is effectively 0)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from datasets.shapenet import ShapeNetDataset
from models.vqvae import VQVAE


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _dict_to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    surface_weight: float = 10.0,
    surface_threshold: float = 0.3,
) -> torch.Tensor:
    """Reconstruction loss, optionally surface-weighted.

    For TSDF (l1/l2): near-surface voxels (|tsdf| < surface_threshold) are
    weighted surface_weight× more heavily than empty-space voxels.
    For occupancy (bce): standard binary cross-entropy, no spatial weighting.
    """
    if mode == "bce":
        return F.binary_cross_entropy(pred, target.float())
    if mode == "l1":
        per_voxel = F.l1_loss(pred, target, reduction="none")
    elif mode in ("l2", "mse"):
        per_voxel = F.mse_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown reconstruction loss: {mode}")

    near_surface = (target.abs() < surface_threshold).float()
    weight = 1.0 + (surface_weight - 1.0) * near_surface
    return (per_voxel * weight).mean()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SimpleNamespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = config.training.warmup_steps
    total = config.training.num_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Latent extraction + stats
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_latents_and_stats(
    model: VQVAE,
    config: SimpleNamespace,
    device: torch.device,
    batch_size: int = 8,
) -> None:
    """Extract compact latents for all splits and save normalization stats.

    Latents are saved per-shape as .npy files. After extraction, mean and std
    over the full training set are computed and written to latent_stats_path
    so the diffusion stage can normalize inputs to ~N(0,1).
    """
    representation = config.representation
    data_root = config.data.root
    latent_dir = Path(data_root) / "latents" / representation
    latent_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = getattr(config.data, "processed_dir", None)

    model.eval()
    for split in ("train", "val", "test"):
        try:
            ds = ShapeNetDataset(
                data_root, split, representation, augment=False,
                processed_dir=processed_dir,
            )
        except (FileNotFoundError, RuntimeError):
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        for x, shape_ids in loader:
            x = x.to(device)
            with autocast("cuda", enabled=config.training.mixed_precision):
                z_c = model.encode_to_latent(x)  # (B, latent_channels, 4, 4, 4)
            for i, sid in enumerate(shape_ids):
                np.save(str(latent_dir / f"{sid}.npy"), z_c[i].cpu().float().numpy())

    print(f"Latents saved to {latent_dir}")

    # Compute normalization stats from training latents only
    train_files = []
    try:
        train_ds = ShapeNetDataset(
            data_root, "train", representation, augment=False,
            processed_dir=processed_dir,
        )
        train_files = [latent_dir / f"{sid}.npy" for sid in train_ds.shape_ids
                       if (latent_dir / f"{sid}.npy").exists()]
    except (FileNotFoundError, RuntimeError):
        train_files = sorted(latent_dir.glob("*.npy"))

    if train_files:
        all_lats = np.stack([np.load(str(f)) for f in train_files])
        lat_mean = float(all_lats.mean())
        lat_std = float(all_lats.std())

        stats_path = Path(getattr(config.diffusion, "latent_stats_path",
                                  str(Path(config.training.checkpoint_dir) / "latent_stats.npz")))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(stats_path), mean=np.array(lat_mean), std=np.array(lat_std))
        print(f"Latent stats: mean={lat_mean:.4f}, std={lat_std:.4f} → {stats_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    representation = config.representation
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs"))

    processed_dir = getattr(config.data, "processed_dir", None)

    train_ds = ShapeNetDataset(
        config.data.root, "train", representation,
        augment=config.data.augment, processed_dir=processed_dir,
    )
    val_ds = ShapeNetDataset(
        config.data.root, "val", representation,
        augment=False, processed_dir=processed_dir,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    print(f"Train: {len(train_ds)} shapes | {steps_per_epoch} steps/epoch")

    model = VQVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler("cuda", enabled=config.training.mixed_precision)

    global_step = 0
    best_val_loss = float("inf")
    epoch_times: list[float] = []   # seconds per epoch
    epoch_start = time.time()
    t_train_start = time.time()

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        state = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        global_step = state["step"]
        best_val_loss = state.get("best_val_loss", best_val_loss)
        epoch_times = state.get("epoch_times", epoch_times)
        print(f"Resumed from step {global_step}")

    loss_cfg = config.loss
    surface_weight = getattr(loss_cfg, "surface_weight", 10.0)
    surface_threshold = getattr(loss_cfg, "surface_threshold", 0.3)
    commit_w = loss_cfg.commitment_weight
    recon_mode = loss_cfg.reconstruction

    model.train()
    data_iter = iter(train_loader)

    while global_step < config.training.num_steps:
        try:
            x, _ = next(data_iter)
        except StopIteration:
            # One epoch completed — record timing
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            writer.add_scalar("train/epoch_time_s", epoch_time, global_step)
            epoch_start = time.time()
            data_iter = iter(train_loader)
            x, _ = next(data_iter)

        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=config.training.mixed_precision):
            out = model(x)
            l_recon = reconstruction_loss(
                out.reconstruction, x, recon_mode, surface_weight, surface_threshold
            )
            # codebook_loss is 0 with EMA; kept in formula for extensibility
            l_total = l_recon + commit_w * out.commitment_loss

        scaler.scale(l_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        if global_step % 100 == 0:
            elapsed = time.time() - t_train_start
            vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0
            print(
                f"step {global_step:6d} | "
                f"recon {l_recon.item():.4f} | "
                f"commit {out.commitment_loss.item():.4f} | "
                f"codebook {out.usage_fraction*100:.1f}% | "
                f"VRAM {vram_gb:.2f}GB | "
                f"{elapsed:.0f}s"
            )
            writer.add_scalar("train/recon_loss",      l_recon.item(),                global_step)
            writer.add_scalar("train/commit_loss",     out.commitment_loss.item(),    global_step)
            writer.add_scalar("train/codebook_usage",  out.usage_fraction,            global_step)
            writer.add_scalar("train/total_loss",      l_total.item(),                global_step)
            writer.add_scalar("train/peak_vram_gb",    vram_gb,                       global_step)

        if global_step % config.training.save_every == 0:
            # Validation pass
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xv, _ in val_loader:
                    xv = xv.to(device, non_blocking=True)
                    with autocast("cuda", enabled=config.training.mixed_precision):
                        out_v = model(xv)
                        lv = reconstruction_loss(
                            out_v.reconstruction, xv, recon_mode, surface_weight, surface_threshold
                        )
                    val_losses.append(lv.item())
            val_loss = float(np.mean(val_losses))
            writer.add_scalar("val/recon_loss", val_loss, global_step)
            print(f"  → val recon loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), str(ckpt_dir / "best.pt"))

            peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0
            avg_epoch_time_s = float(np.mean(epoch_times)) if epoch_times else 0.0
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": global_step,
                "best_val_loss": best_val_loss,
                "epoch_times": epoch_times,
                "peak_vram_gb": peak_vram_gb,
                "avg_epoch_time_s": avg_epoch_time_s,
            }
            torch.save(state, str(ckpt_dir / "latest.pt"))
            torch.save(state, str(ckpt_dir / f"step_{global_step:06d}.pt"))
            model.train()

    writer.close()
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0
    avg_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0
    print(f"Training complete.")
    print(f"  Peak VRAM         : {peak_vram_gb:.2f} GB")
    print(f"  Avg time/epoch    : {avg_epoch_s:.1f}s")

    print("Extracting and saving latents...")
    save_latents_and_stats(model, config, device, batch_size=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)
