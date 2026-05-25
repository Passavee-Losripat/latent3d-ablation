"""Convert a binary voxel grid to a truncated signed distance field (TSDF).

Input:  (32, 32, 32) uint8 binary voxel grid (1=occupied, 0=empty)
Output: (32, 32, 32) float32 TSDF array with values in [-1.0, 1.0], saved as .npy

Algorithm:
  1. EDT from occupied voxels  → distance_outside (positive outside surface)
  2. EDT from empty voxels      → distance_inside  (positive inside surface)
  3. SDF = distance_outside - distance_inside
  4. Clamp to [-truncation, +truncation], then normalize to [-1, 1]
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from pathlib import Path


def voxel_to_tsdf(voxel: np.ndarray, truncation: float = 3.0) -> np.ndarray:
    """Compute a normalized TSDF from a binary voxel grid.

    Args:
        voxel:      (32, 32, 32) uint8 array, 1=occupied.
        truncation: truncation distance in voxel units.

    Returns:
        (32, 32, 32) float32 array in [-1.0, 1.0].
    """
    occupied = voxel.astype(bool)
    empty = ~occupied

    # Distance from each empty voxel to the nearest occupied voxel (outside distance)
    dist_outside = distance_transform_edt(empty).astype(np.float32)
    # Distance from each occupied voxel to the nearest empty voxel (inside distance)
    dist_inside = distance_transform_edt(occupied).astype(np.float32)

    # SDF: positive outside, negative inside
    sdf = np.where(occupied, -dist_inside, dist_outside)

    # Truncate and normalize to [-1, 1]
    sdf = np.clip(sdf, -truncation, truncation)
    tsdf = sdf / truncation  # now in [-1, 1]
    return tsdf.astype(np.float32)


def compute_and_save_tsdf(
    voxel: np.ndarray,
    tsdf_out_path: str | Path,
    truncation: float = 3.0,
) -> np.ndarray:
    """Compute TSDF from binary voxel grid and save to disk."""
    tsdf = voxel_to_tsdf(voxel, truncation)
    Path(tsdf_out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(tsdf_out_path), tsdf)
    return tsdf
