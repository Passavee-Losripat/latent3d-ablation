"""Evaluation script: computes all Stage 1 (reconstruction) and Stage 2 (generation) metrics.

Stage 1 — reconstruction (run after VQ-VAE training):
  Chamfer Distance (CD)   — reconstructed vs GT point clouds via marching cubes
  IoU                     — thresholded TSDF binary vs GT occupancy
  F-Score @ 1%            — precision/recall at 1% distance threshold

Stage 2 — generation (run after diffusion training):
  MMD                     — minimum matching distance (generated vs test set)
  Coverage (COV)          — fraction of test shapes matched by ≥1 generated shape
  JSD                     — Jensen-Shannon divergence of voxel occupancy distributions

Compute profiling (reported for every run):
  Peak VRAM (GB)          — torch.cuda.max_memory_allocated
  Training time/epoch     — loaded from VQ-VAE checkpoint (logged during training)
  Inference time/shape    — timed during Stage 2 generation

Usage:
    python evaluate.py --config configs/tsdf.yaml \\
                       --vqvae_ckpt checkpoints/tsdf/best.pt \\
                       [--stage {1,2,both}] \\
                       [--unet_ckpt checkpoints/tsdf_diffusion/latest.pt]
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.amp import autocast
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
        return None
    verts, _, _, _ = marching_cubes(tsdf, level=threshold)
    verts = verts / np.array(tsdf.shape)   # normalize to [0, 1]
    return verts.astype(np.float32)


def sample_point_cloud(verts: np.ndarray, n: int = 2048) -> np.ndarray:
    """Subsample or upsample point cloud to exactly n points."""
    idx = np.random.choice(len(verts), n, replace=(len(verts) < n))
    return verts[idx]


# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Symmetric Chamfer distance between two (N,3) and (M,3) point clouds."""
    p1 = torch.from_numpy(pc1).float()  # (N, 3)
    p2 = torch.from_numpy(pc2).float()  # (M, 3)
    dists = torch.cdist(p1, p2)         # (N, M)
    cd = dists.min(1).values.mean() + dists.min(0).values.mean()
    return cd.item()


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_from_tsdf(
    pred_tsdf: np.ndarray,
    gt_voxel: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Binary IoU: threshold prediction → compare with GT binary voxel.

    For TSDF/triplane (threshold=0.0): inside where pred <= 0.
    For occupancy (threshold=0.5):     inside where pred >= 0.5.
    """
    if threshold == 0.5:
        pred_bin = (pred_tsdf >= threshold).astype(np.float32)
    else:
        pred_bin = (pred_tsdf <= threshold).astype(np.float32)
    gt_bin   = (gt_voxel  >  0).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    union        = ((pred_bin + gt_bin) > 0).sum()
    return float(intersection / (union + 1e-8))


# ---------------------------------------------------------------------------
# F-Score
# ---------------------------------------------------------------------------

def fscore(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    threshold: float = 0.01,
) -> tuple[float, float, float]:
    """F-Score, precision, and recall at given distance threshold."""
    p1 = torch.from_numpy(pc_pred).float()
    p2 = torch.from_numpy(pc_gt).float()
    dists = torch.cdist(p1, p2)
    precision = (dists.min(1).values < threshold).float().mean().item()
    recall    = (dists.min(0).values < threshold).float().mean().item()
    f = 2 * precision * recall / (precision + recall + 1e-8)
    return f, precision, recall


# ---------------------------------------------------------------------------
# Stage 2 metrics
# ---------------------------------------------------------------------------

def _pc_to_voxel(pc: np.ndarray, res: int = 64) -> np.ndarray:
    """Rasterize point cloud to binary voxel grid for JSD."""
    coords = np.clip((pc * res).astype(int), 0, res - 1)
    vox = np.zeros((res, res, res), dtype=np.float32)
    vox[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return vox


def mmd_coverage(
    gen_pcs: list[np.ndarray],
    ref_pcs: list[np.ndarray],
) -> tuple[float, float]:
    """Minimum matching distance and coverage (fraction of test shapes covered)."""
    min_dists = []
    matched: set[int] = set()
    for g in gen_pcs:
        best, best_j = float("inf"), -1
        for j, r in enumerate(ref_pcs):
            d = chamfer_distance(g, r)
            if d < best:
                best, best_j = d, j
        min_dists.append(best)
        matched.add(best_j)
    return float(np.mean(min_dists)), len(matched) / len(ref_pcs)


def jsd_voxel(
    gen_pcs: list[np.ndarray],
    ref_pcs: list[np.ndarray],
    res: int = 64,
) -> float:
    """Jensen-Shannon divergence of marginal voxel occupancy distributions."""
    from scipy.spatial.distance import jensenshannon

    gen_occ = np.mean([_pc_to_voxel(p, res) for p in gen_pcs], axis=0).flatten()
    ref_occ = np.mean([_pc_to_voxel(p, res) for p in ref_pcs], axis=0).flatten()
    gen_occ = gen_occ / (gen_occ.sum() + 1e-8)
    ref_occ = ref_occ / (ref_occ.sum() + 1e-8)
    return float(jensenshannon(gen_occ, ref_occ))


# ---------------------------------------------------------------------------
# Latent stats helpers
# ---------------------------------------------------------------------------

def _load_latent_stats(diff_cfg: SimpleNamespace) -> tuple[float, float]:
    """Load normalization stats; fall back to (0, 1) if not found."""
    stats_path = getattr(diff_cfg, "latent_stats_path", None)
    if stats_path and Path(stats_path).exists():
        data = np.load(str(stats_path))
        return float(data["mean"]), float(data["std"])
    return 0.0, 1.0


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
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=False))
    vqvae.eval()

    representation = config.representation
    processed_dir  = getattr(config.data, "processed_dir", None)
    voxel_dir      = Path(getattr(config.data, "voxel_dir",
                                  str(Path(config.data.root) / "processed" / "voxel_64")))

    test_ds = ShapeNetDataset(
        config.data.root, "test", representation, augment=False,
        processed_dir=processed_dir,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    chamfers, ious, fscores = [], [], []
    mixed = config.training.mixed_precision
    iso_threshold = getattr(config.evaluation, "iso_threshold", 0.0)

    for x, (shape_id,) in loader:
        x = x.to(device)
        with torch.no_grad(), autocast("cuda", enabled=mixed):
            out = vqvae(x)
        pred_raw = out.reconstruction[0, 0].cpu().float()
        # Occupancy decoder outputs logits — apply sigmoid before iso-surface extraction
        if iso_threshold == 0.5:
            pred_raw = torch.sigmoid(pred_raw)
        pred_tsdf = pred_raw.numpy()
        gt_tsdf   = x[0, 0].cpu().float().numpy()

        pred_pc = tsdf_to_pointcloud(pred_tsdf, threshold=iso_threshold)
        gt_pc   = tsdf_to_pointcloud(gt_tsdf,   threshold=iso_threshold)
        if pred_pc is not None and gt_pc is not None and len(pred_pc) > 0 and len(gt_pc) > 0:
            p = sample_point_cloud(pred_pc, n_points)
            g = sample_point_cloud(gt_pc,   n_points)
            chamfers.append(chamfer_distance(p, g))
            f, _, _ = fscore(p, g, threshold=config.evaluation.fscore_threshold)
            fscores.append(f)

        vox_path = voxel_dir / f"{shape_id}.npy"
        if vox_path.exists():
            gt_voxel = np.load(str(vox_path))
            ious.append(iou_from_tsdf(pred_tsdf, gt_voxel, threshold=iso_threshold))

    return {
        "chamfer_distance":  float(np.mean(chamfers)) if chamfers else float("nan"),
        "iou":               float(np.mean(ious))     if ious     else float("nan"),
        "fscore_at_1pct":    float(np.mean(fscores))  if fscores  else float("nan"),
        "n_shapes":          len(test_ds),
    }


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
    resolution = getattr(config.model, "resolution", 64)
    mixed = config.training.mixed_precision

    # Load latent normalization stats
    lat_mean, lat_std = _load_latent_stats(diff_cfg)
    print(f"Latent stats: mean={lat_mean:.4f}, std={lat_std:.4f}")

    # Load VQ-VAE
    vqvae = VQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device, weights_only=False))
    vqvae.eval()

    # Load diffusion checkpoint — auto-detect in_channels from saved state
    diff_state = torch.load(unet_ckpt, map_location=device, weights_only=False)
    in_channels = diff_state.get("in_channels", None)
    if in_channels is None:
        # Fall back: infer from any saved latent file
        latent_files = sorted(Path(diff_cfg.latent_dir).glob("*.npy"))
        if latent_files:
            in_channels = int(np.load(str(latent_files[0])).shape[0])
        else:
            raise RuntimeError("Cannot determine latent in_channels: no latent files found.")

    ddpm = DDPM(timesteps=diff_cfg.timesteps).to(device)
    unet = UNet3D(in_channels=in_channels).to(device)
    unet.load_state_dict(diff_state["unet"])
    unet.eval()

    sampler = DDIMSampler(ddpm, ddim_steps=diff_cfg.ddim_steps)

    # Reference test set point clouds
    processed_dir = getattr(config.data, "processed_dir", None)
    test_ds = ShapeNetDataset(
        config.data.root, "test", config.representation, augment=False,
        processed_dir=processed_dir,
    )
    n_points_eval = getattr(config.evaluation, "n_points", n_points)
    ref_pcs: list[np.ndarray] = []
    for i in range(min(len(test_ds), 200)):
        x, _ = test_ds[i]
        pc = tsdf_to_pointcloud(x[0].numpy())
        if pc is not None and len(pc) > 0:
            ref_pcs.append(sample_point_cloud(pc, n_points_eval))

    # Generate shapes and time inference
    gen_pcs: list[np.ndarray] = []
    batch_size = config.training.batch_size
    shape_per_latent = (in_channels, 4, 4, 4)

    t0 = time.time()
    while len(gen_pcs) < n_generate:
        bs = min(batch_size, n_generate - len(gen_pcs))
        z = sampler.sample(unet, (bs, *shape_per_latent), device)
        # Denormalize before decoding
        z = z * lat_std + lat_mean
        with torch.no_grad(), autocast("cuda", enabled=mixed):
            tsdf_batch = vqvae.decode_latent(z)
        for i in range(bs):
            tsdf = tsdf_batch[i, 0].cpu().float().numpy()
            pc   = tsdf_to_pointcloud(tsdf)
            if pc is not None and len(pc) > 0:
                gen_pcs.append(sample_point_cloud(pc, n_points_eval))

    inference_time_per_shape = (time.time() - t0) / max(1, len(gen_pcs))

    if not gen_pcs or not ref_pcs:
        return {"error": "insufficient valid point clouds for Stage 2 metrics"}

    mmd, cov = mmd_coverage(gen_pcs, ref_pcs)
    jsd = jsd_voxel(gen_pcs, ref_pcs, res=resolution)

    return {
        "mmd":                       mmd,
        "coverage":                  cov,
        "jsd":                       jsd,
        "n_generated":               len(gen_pcs),
        "inference_time_per_shape_s": inference_time_per_shape,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--unet_ckpt",  default=None)
    parser.add_argument("--stage",      choices=["1", "2", "both"], default="1")
    parser.add_argument("--n_generate", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Read training profiling stats — best.pt only has weights, so also try latest.pt
    vqvae_state = torch.load(args.vqvae_ckpt, map_location="cpu", weights_only=False)
    avg_epoch_time_s = vqvae_state.get("avg_epoch_time_s", None)
    ckpt_peak_vram   = vqvae_state.get("peak_vram_gb",     None)
    if avg_epoch_time_s is None:
        latest = Path(args.vqvae_ckpt).parent / "latest.pt"
        if latest.exists():
            s = torch.load(str(latest), map_location="cpu", weights_only=False)
            avg_epoch_time_s = s.get("avg_epoch_time_s", None)
            ckpt_peak_vram   = s.get("peak_vram_gb", ckpt_peak_vram)

    if args.stage in ("1", "both"):
        print("─" * 50)
        print("Stage 1: Reconstruction")
        print("─" * 50)
        t0 = time.time()
        eval_cfg = config.evaluation
        r1 = evaluate_stage1(
            config, args.vqvae_ckpt, device,
            n_points=getattr(eval_cfg, "n_points", 2048),
        )
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

        print(f"Chamfer Distance (CD) : {r1['chamfer_distance']:.6f}")
        print(f"IoU                  : {r1['iou']:.4f}")
        print(f"F-Score @ 1%         : {r1['fscore_at_1pct']:.4f}")
        print(f"─── Compute Profiling ───")
        print(f"Peak VRAM            : {peak_vram_gb:.2f} GB"
              + (f"  (training peak: {ckpt_peak_vram:.2f} GB)" if ckpt_peak_vram else ""))
        if avg_epoch_time_s is not None:
            print(f"Training time/epoch  : {avg_epoch_time_s:.1f}s")
        else:
            print(f"Training time/epoch  : N/A (run train_vqvae.py to log)")
        print(f"Eval time            : {time.time()-t0:.1f}s over {r1['n_shapes']} shapes")

    if args.stage in ("2", "both"):
        if args.unet_ckpt is None:
            print("\n--unet_ckpt required for Stage 2 evaluation.")
        else:
            print("\n" + "─" * 50)
            print("Stage 2: Generation")
            print("─" * 50)
            r2 = evaluate_stage2(
                config, args.vqvae_ckpt, args.unet_ckpt, device,
                n_generate=args.n_generate,
                n_points=getattr(config.evaluation, "n_points", 2048),
            )
            if "error" in r2:
                print(f"Error: {r2['error']}")
            else:
                print(f"MMD                  : {r2['mmd']:.6f}")
                print(f"Coverage (COV)       : {r2['coverage']:.4f}")
                print(f"JSD                  : {r2['jsd']:.6f}")
                print(f"─── Compute Profiling ───")
                inf_time = r2['inference_time_per_shape_s']
                print(f"Inference time/shape : {inf_time:.2f}s")
                print(f"N generated          : {r2['n_generated']}")


if __name__ == "__main__":
    main()
