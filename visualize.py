"""Visualize generated or reconstructed TSDF shapes as PNG images.

Produces two outputs:
  - slices.png   : mid-plane TSDF slice grid (fast, no extra deps)
  - surfaces.png : 3D marching-cubes mesh renders (requires scikit-image)

Usage:
    # Visualize generated shapes (after train_diffusion.py)
    python visualize.py --source outputs/tsdf/ --out vis/

    # Visualize reconstructions (after train_vqvae.py)
    python visualize.py --source outputs/tsdf/ --out vis/ --title "Reconstructions"

    # Generate on the fly from a trained model (no pre-saved files needed)
    python visualize.py \
        --config configs/tsdf.yaml \
        --vqvae_ckpt checkpoints/tsdf/best.pt \
        --unet_ckpt  checkpoints/tsdf_diffusion/latest.pt \
        --n_gen 16 --out vis/
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed on server
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tsdfs(source_dir: Path, limit: int) -> list[np.ndarray]:
    files = sorted(source_dir.glob("*.npy"))[:limit]
    if not files:
        raise RuntimeError(f"No .npy files found in {source_dir}")
    return [np.load(str(f)).squeeze() for f in files]   # squeeze (1,R,R,R) → (R,R,R)


def save_slice_grid(tsdfs: list[np.ndarray], out_path: Path, title: str) -> None:
    """Save a grid of mid-plane TSDF slices (red=outside, blue=inside)."""
    n = len(tsdfs)
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i, tsdf in enumerate(tsdfs):
        mid = tsdf.shape[2] // 2
        axes[i].imshow(tsdf[:, :, mid], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[i].axis("off")
        axes[i].set_title(f"#{i}", fontsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved slice grid  → {out_path}")


def save_surface_grid(tsdfs: list[np.ndarray], out_path: Path, title: str) -> None:
    """Save a grid of 3D surface renders via marching cubes."""
    try:
        from skimage.measure import marching_cubes
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("scikit-image not installed — skipping 3D surface render.")
        return

    n = len(tsdfs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    res = tsdfs[0].shape[0]

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=10)

    for i, tsdf in enumerate(tsdfs):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        try:
            verts, faces, _, _ = marching_cubes(tsdf, level=0.0)
            mesh = Poly3DCollection(
                verts[faces], alpha=0.8,
                facecolor="steelblue", edgecolor="none"
            )
            ax.add_collection3d(mesh)
            ax.set_xlim(0, res); ax.set_ylim(0, res); ax.set_zlim(0, res)
        except Exception:
            ax.text(0.5, 0.5, 0.5, "no surface", ha="center", transform=ax.transAxes)
        ax.set_title(f"#{i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved surface grid → {out_path}")


# ---------------------------------------------------------------------------
# On-the-fly generation
# ---------------------------------------------------------------------------

def generate_tsdfs(
    config_path: str,
    vqvae_ckpt: str,
    unet_ckpt: str,
    n_gen: int,
    out_dir: Path,
) -> list[np.ndarray]:
    import torch
    import yaml
    from torch.amp import autocast

    def _dict_to_ns(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
        return ns

    with open(config_path) as f:
        config = _dict_to_ns(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.vqvae import VQVAE
    from diffusion.ddpm import DDPM
    from diffusion.unet3d import UNet3D
    from diffusion.ddim import DDIMSampler

    vqvae = VQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=False))
    vqvae.eval()

    diff_state = torch.load(unet_ckpt, map_location=device, weights_only=False)
    in_channels = diff_state.get("in_channels", 64)
    lat_mean    = diff_state.get("lat_mean",    0.0)
    lat_std     = diff_state.get("lat_std",     1.0)

    ddpm = DDPM(timesteps=config.diffusion.timesteps).to(device)
    unet = UNet3D(in_channels=in_channels).to(device)
    unet.load_state_dict(diff_state["unet"])
    unet.eval()

    sampler = DDIMSampler(ddpm, ddim_steps=config.diffusion.ddim_steps)
    mixed   = config.training.mixed_precision

    tsdfs = []
    batch_size = config.training.batch_size
    print(f"Generating {n_gen} shapes via DDIM ({config.diffusion.ddim_steps} steps)...")

    while len(tsdfs) < n_gen:
        bs = min(batch_size, n_gen - len(tsdfs))
        z  = sampler.sample(unet, (bs, in_channels, 4, 4, 4), device)
        z  = z * lat_std + lat_mean
        with torch.no_grad(), autocast("cuda", enabled=mixed):
            batch = vqvae.decode_latent(z)
        for i in range(bs):
            tsdf = batch[i, 0].cpu().float().numpy()
            tsdfs.append(tsdf)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_dir / f"generated_{len(tsdfs)-1:04d}.npy"), tsdf)

    return tsdfs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     type=str, default=None,
                        help="Directory of pre-saved .npy TSDF files.")
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--vqvae_ckpt", type=str, default=None)
    parser.add_argument("--unet_ckpt",  type=str, default=None)
    parser.add_argument("--n_gen",      type=int, default=16)
    parser.add_argument("--out",        type=str, default="vis/")
    parser.add_argument("--title",      type=str, default="Generated Chairs (TSDF)")
    parser.add_argument("--no_surface", action="store_true",
                        help="Skip the 3D surface render (faster).")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.source:
        tsdfs = load_tsdfs(Path(args.source), limit=args.n_gen)
    elif args.config and args.vqvae_ckpt and args.unet_ckpt:
        tsdfs = generate_tsdfs(
            args.config, args.vqvae_ckpt, args.unet_ckpt,
            args.n_gen, out_dir / "npy"
        )
    else:
        parser.error("Provide either --source or --config + --vqvae_ckpt + --unet_ckpt")

    print(f"Visualizing {len(tsdfs)} shapes...")
    save_slice_grid(tsdfs, out_dir / "slices.png",   title=args.title)
    if not args.no_surface:
        save_surface_grid(tsdfs, out_dir / "surfaces.png", title=args.title)


if __name__ == "__main__":
    main()
