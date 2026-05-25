"""Evaluation script: computes Stage 1 (reconstruction) and Stage 2 (generation) metrics.

Stage 1 metrics (run after VQ-VAE training):
  - Chamfer Distance  — reconstructed vs GT point clouds via marching cubes
  - IoU               — thresholded TSDF binary vs GT occupancy
  - F-Score @ 1%      — precision/recall at 1% distance threshold

Stage 2 metrics (run after diffusion training, on generated shapes):
  - MMD  — minimum matching distance (generated vs test set)
  - Coverage — fraction of test shapes matched by ≥1 generated shape
  - JSD  — Jensen-Shannon divergence of voxel occupancy distributions

Compute profiling is also logged:
  - Peak VRAM, training time per epoch, inference time per shape

Usage:
    python evaluate.py --config configs/tsdf.yaml \
                       --vqvae_ckpt checkpoints/tsdf/best.pt \
                       [--stage {1,2,both}] \
                       [--unet_ckpt checkpoints/tsdf_diffusion/latest.pt]
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from datasets.shapenet import ShapeNetDataset
from models.vqvae import VQVAE
from diffusion.ddpm import DDPM
from diffusion.unet3d import UNet3D
from diffusion.ddim import DDIMSampler


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
# Point cloud helpers
# ---------------------------------------------------------------------------

def tsdf_to_pointcloud(tsdf: np.ndarray, threshold: float = 0.0) -> np.ndarray | None:
    """Extract surface point cloud from a TSDF using marching cubes."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("pip install scikit-image for mesh extraction.")

    if tsdf.max() <= threshold or tsdf.min() >= threshold:
        return None  # no iso-surface
    verts, _, _, _ = marching_cubes(tsdf, level=threshold)
    # Normalize to [0, 1]
    verts = verts / np.array(tsdf.shape)
    return verts.astype(np.float32)


def sample_point_cloud(verts: np.ndarray, n: int = 2048) -> np.ndarray:
    """Subsample or upsample point cloud to exactly n points."""
    if len(verts) >= n:
        idx = np.random.choice(len(verts), n, replace=False)
    else:
        idx = np.random.choice(len(verts), n, replace=True)
    return verts[idx]


# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds (N,3) and (M,3)."""
    p1 = torch.from_numpy(pc1).float().unsqueeze(0)  # (1, N, 3)
    p2 = torch.from_numpy(pc2).float().unsqueeze(0)  # (1, M, 3)

    diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # (1, N, M, 3)
    dists = diff.pow(2).sum(-1)  # (1, N, M)

    cd = dists.min(2).values.mean() + dists.min(1).values.mean()
    return cd.item()


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_from_tsdf(
    pred_tsdf: np.ndarray,
    gt_voxel: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Binary IoU: threshold predicted TSDF at 0 → compare with GT binary voxel."""
    pred_bin = (pred_tsdf <= threshold).astype(np.float32)
    gt_bin = (gt_voxel > 0).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    union = ((pred_bin + gt_bin) > 0).sum()
    return float(intersection / (union + 1e-8))


# ---------------------------------------------------------------------------
# F-Score
# ---------------------------------------------------------------------------

def fscore(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    threshold: float = 0.01,
) -> tuple[float, float, float]:
    """F-Score at given distance threshold."""
    p1 = torch.from_numpy(pc_pred).float()
    p2 = torch.from_numpy(pc_gt).float()

    diffs_12 = torch.cdist(p1, p2)  # (N, M)
    diffs_21 = diffs_12.t()

    precision = (diffs_12.min(1).values < threshold).float().mean().item()
    recall = (diffs_21.min(1).values < threshold).float().mean().item()
    f = 2 * precision * recall / (precision + recall + 1e-8)
    return f, precision, recall


# ---------------------------------------------------------------------------
# Stage 2 metrics
# ---------------------------------------------------------------------------

def _pc_to_voxel(pc: np.ndarray, res: int = 32) -> np.ndarray:
    """Rasterize point cloud to binary voxel grid for JSD."""
    coords = np.clip((pc * res).astype(int), 0, res - 1)
    vox = np.zeros((res, res, res), dtype=np.float32)
    vox[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return vox


def mmd_coverage(
    gen_pcs: list[np.ndarray],
    ref_pcs: list[np.ndarray],
) -> tuple[float, float]:
    """Minimum matching distance and coverage."""
    min_dists = []
    matched = set()
    for g in gen_pcs:
        best = float("inf")
        best_j = -1
        for j, r in enumerate(ref_pcs):
            d = chamfer_distance(g, r)
            if d < best:
                best, best_j = d, j
        min_dists.append(best)
        matched.add(best_j)
    mmd = float(np.mean(min_dists))
    coverage = len(matched) / len(ref_pcs)
    return mmd, coverage


def jsd_voxel(
    gen_pcs: list[np.ndarray],
    ref_pcs: list[np.ndarray],
    res: int = 32,
) -> float:
    """Jensen-Shannon divergence of voxel occupancy marginals."""
    from scipy.spatial.distance import jensenshannon

    gen_occ = np.mean([_pc_to_voxel(p, res) for p in gen_pcs], axis=0).flatten()
    ref_occ = np.mean([_pc_to_voxel(p, res) for p in ref_pcs], axis=0).flatten()
    # Normalize to probability distributions
    gen_occ = gen_occ / (gen_occ.sum() + 1e-8)
    ref_occ = ref_occ / (ref_occ.sum() + 1e-8)
    return float(jensenshannon(gen_occ, ref_occ))


# ---------------------------------------------------------------------------
# Stage 1 evaluation
# ---------------------------------------------------------------------------

def evaluate_stage1(
    config: SimpleNamespace,
    vqvae_ckpt: str,
    device: torch.device,
    n_points: int = 2048,
) -> dict[str, float]:
    vqvae = VQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae.eval()

    representation = config.representation
    test_ds = ShapeNetDataset(config.data.root, "test", representation, augment=False)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # We also need the binary voxel grids for IoU
    voxel_dir = Path(config.data.root) / "processed" / "voxel"

    chamfers, ious, fscores = [], [], []

    for x, (shape_id,) in loader:
        x = x.to(device)
        with torch.no_grad(), autocast():
            out = vqvae(x)
        pred_tsdf = out.reconstruction[0, 0].cpu().float().numpy()
        gt_tsdf = x[0, 0].cpu().float().numpy()

        # Chamfer
        pred_pc = tsdf_to_pointcloud(pred_tsdf)
        gt_pc = tsdf_to_pointcloud(gt_tsdf)
        if pred_pc is not None and gt_pc is not None:
            p_pc = sample_point_cloud(pred_pc, n_points)
            g_pc = sample_point_cloud(gt_pc, n_points)
            chamfers.append(chamfer_distance(p_pc, g_pc))
            f, _, _ = fscore(p_pc, g_pc, threshold=config.evaluation.fscore_threshold)
            fscores.append(f)

        # IoU vs binary voxel
        vox_path = voxel_dir / f"{shape_id}.npy"
        if vox_path.exists():
            gt_voxel = np.load(str(vox_path))
            ious.append(iou_from_tsdf(pred_tsdf, gt_voxel))

    results = {
        "chamfer_mean": float(np.mean(chamfers)) if chamfers else float("nan"),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "fscore_mean": float(np.mean(fscores)) if fscores else float("nan"),
        "n_shapes": len(loader),
    }
    return results


# ---------------------------------------------------------------------------
# Stage 2 evaluation
# ---------------------------------------------------------------------------

def evaluate_stage2(
    config: SimpleNamespace,
    vqvae_ckpt: str,
    unet_ckpt: str,
    device: torch.device,
    n_generate: int = 100,
    n_points: int = 2048,
) -> dict[str, float]:
    diff_cfg = config.diffusion

    # Load models
    vqvae = VQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae.eval()

    ddpm = DDPM(timesteps=diff_cfg.timesteps).to(device)
    unet = UNet3D(in_channels=256).to(device)
    state = torch.load(unet_ckpt, map_location=device)
    unet.load_state_dict(state["unet"])
    unet.eval()

    sampler = DDIMSampler(ddpm, ddim_steps=diff_cfg.ddim_steps)

    # Reference test set point clouds
    test_ds = ShapeNetDataset(config.data.root, "test", config.representation, augment=False)
    ref_pcs: list[np.ndarray] = []
    for i in range(min(len(test_ds), 200)):
        x, _ = test_ds[i]
        tsdf = x[0].numpy()
        pc = tsdf_to_pointcloud(tsdf)
        if pc is not None:
            ref_pcs.append(sample_point_cloud(pc, n_points))

    # Generate shapes
    gen_pcs: list[np.ndarray] = []
    batch_size = config.training.batch_size
    latent_sample = np.load(str(next(Path(diff_cfg.latent_dir).glob("*.npy"))))
    c, h, w, l = latent_sample.shape

    t0 = time.time()
    n_batches = (n_generate + batch_size - 1) // batch_size
    for _ in range(n_batches):
        bs = min(batch_size, n_generate - len(gen_pcs))
        z = sampler.sample(unet, (bs, c, h, w, l), device)
        with torch.no_grad():
            tsdf_batch = vqvae.decode_latent(z)
        for i in range(bs):
            tsdf = tsdf_batch[i, 0].cpu().numpy()
            pc = tsdf_to_pointcloud(tsdf)
            if pc is not None:
                gen_pcs.append(sample_point_cloud(pc, n_points))
        if len(gen_pcs) >= n_generate:
            break

    inference_time = (time.time() - t0) / max(1, len(gen_pcs))

    if not gen_pcs or not ref_pcs:
        return {"error": "insufficient valid point clouds"}

    mmd, cov = mmd_coverage(gen_pcs, ref_pcs)
    jsd = jsd_voxel(gen_pcs, ref_pcs)

    return {
        "mmd": mmd,
        "coverage": cov,
        "jsd": jsd,
        "n_generated": len(gen_pcs),
        "inference_time_per_shape_s": inference_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--unet_ckpt", default=None)
    parser.add_argument("--stage", choices=["1", "2", "both"], default="1")
    parser.add_argument("--n_generate", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.stage in ("1", "both"):
        print("\n--- Stage 1: Reconstruction ---")
        t0 = time.time()
        r1 = evaluate_stage1(config, args.vqvae_ckpt, device)
        print(f"Chamfer Distance : {r1['chamfer_mean']:.6f}")
        print(f"IoU              : {r1['iou_mean']:.4f}")
        print(f"F-Score          : {r1['fscore_mean']:.4f}")
        print(f"Peak VRAM        : {torch.cuda.max_memory_allocated(device)/1e6:.0f} MB")
        print(f"Eval time        : {time.time()-t0:.1f}s over {r1['n_shapes']} shapes")

    if args.stage in ("2", "both"):
        if args.unet_ckpt is None:
            print("--unet_ckpt required for Stage 2 evaluation.")
        else:
            print("\n--- Stage 2: Generation ---")
            r2 = evaluate_stage2(
                config, args.vqvae_ckpt, args.unet_ckpt, device, n_generate=args.n_generate
            )
            for k, v in r2.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
