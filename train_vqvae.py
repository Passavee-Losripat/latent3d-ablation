"""Stage 1 training: VQ-VAE on 3D voxel representations.

Usage:
    python train_vqvae.py --config configs/tsdf.yaml

After training, saves encoder+VQ latents for every training shape to
data/latents/{representation}/<shape_id>.npy for diffusion stage 2.

Total loss:  L = L_recon + vq_weight * L_codebook + commitment_weight * L_commit
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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

def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l1":
        return F.l1_loss(pred, target)
    if mode == "l2" or mode == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown reconstruction loss: {mode}")


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
        # cosine decay from 1.0 → 0.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_latents(
    model: VQVAE,
    data_root: str,
    representation: str,
    device: torch.device,
    batch_size: int = 8,
) -> None:
    """Run all training shapes through encoder+VQ and save latents as .npy."""
    from datasets.shapenet import ShapeNetDataset

    latent_dir = Path(data_root) / "latents" / representation
    latent_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for split in ("train", "val", "test"):
        try:
            ds = ShapeNetDataset(data_root, split, representation, augment=False)
        except (FileNotFoundError, RuntimeError):
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        for x, shape_ids in loader:
            x = x.to(device)
            with autocast():
                z_q = model.encode_to_latent(x)  # (B, 256, 4, 4, 4)
            for i, sid in enumerate(shape_ids):
                np.save(str(latent_dir / f"{sid}.npy"), z_q[i].cpu().float().numpy())

    print(f"Latents saved to {latent_dir}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    representation = config.representation
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs"))

    # Dataset & loader
    train_ds = ShapeNetDataset(
        config.data.root, "train", representation, augment=config.data.augment
    )
    val_ds = ShapeNetDataset(
        config.data.root, "val", representation, augment=False
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

    # Model
    model = VQVAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=config.training.mixed_precision)

    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    # Resume from latest checkpoint if available
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        state = torch.load(str(latest_ckpt), map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        global_step = state["step"]
        best_val_loss = state.get("best_val_loss", best_val_loss)
        print(f"Resumed from step {global_step}")

    vq_w = config.loss.vq_weight
    commit_w = config.loss.commitment_weight
    recon_mode = config.loss.reconstruction

    model.train()
    data_iter = iter(train_loader)

    while global_step < config.training.num_steps:
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, _ = next(data_iter)

        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.training.mixed_precision):
            out = model(x)
            l_recon = reconstruction_loss(out.reconstruction, x, recon_mode)
            l_total = l_recon + vq_w * out.codebook_loss + commit_w * out.commitment_loss

        scaler.scale(l_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        if global_step % 100 == 0:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0
            print(
                f"step {global_step:6d} | "
                f"recon {l_recon.item():.4f} | "
                f"vq {out.codebook_loss.item():.4f} | "
                f"commit {out.commitment_loss.item():.4f} | "
                f"codebook {out.usage_fraction*100:.1f}% | "
                f"VRAM {vram_mb:.0f}MB | "
                f"{elapsed:.0f}s"
            )
            writer.add_scalar("train/recon_loss", l_recon.item(), global_step)
            writer.add_scalar("train/codebook_loss", out.codebook_loss.item(), global_step)
            writer.add_scalar("train/commit_loss", out.commitment_loss.item(), global_step)
            writer.add_scalar("train/codebook_usage", out.usage_fraction, global_step)
            writer.add_scalar("train/total_loss", l_total.item(), global_step)

        if global_step % config.training.save_every == 0:
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xv, _ in val_loader:
                    xv = xv.to(device, non_blocking=True)
                    with autocast(enabled=config.training.mixed_precision):
                        out_v = model(xv)
                        lv = reconstruction_loss(out_v.reconstruction, xv, recon_mode)
                    val_losses.append(lv.item())
            val_loss = float(np.mean(val_losses))
            writer.add_scalar("val/recon_loss", val_loss, global_step)
            print(f"  → val recon loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), str(ckpt_dir / "best.pt"))

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": global_step,
                "best_val_loss": best_val_loss,
            }
            torch.save(state, str(ckpt_dir / "latest.pt"))
            torch.save(state, str(ckpt_dir / f"step_{global_step:06d}.pt"))
            model.train()

    writer.close()
    print(f"Training complete. Peak VRAM: {torch.cuda.max_memory_allocated(device)/1e6:.0f}MB")

    # Save latents for diffusion stage
    print("Extracting and saving latents...")
    save_latents(model, config.data.root, representation, device, batch_size=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)
