"""Voxelization of raw .obj meshes into (32, 32, 32) binary numpy arrays.

Input:  path to a .obj mesh file
Output: (32, 32, 32) uint8 numpy array, 1=occupied 0=empty, saved as .npy
"""

import numpy as np
import trimesh
from pathlib import Path


def voxelize_mesh(obj_path: str | Path, resolution: int = 32) -> np.ndarray:
    """Load an .obj mesh and return a binary voxel grid of shape (resolution,)*3."""
    mesh = trimesh.load(str(obj_path), force='mesh')

    # Normalize mesh to unit cube centered at origin
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    scale = (mesh.bounds[1] - mesh.bounds[0]).max()
    if scale > 0:
        mesh.apply_scale(1.0 / scale)

    voxel_grid = mesh.voxelized(pitch=1.0 / resolution)
    voxel_grid = voxel_grid.fill()  # fill interior

    # Convert to dense array and crop/pad to exact resolution
    dense = voxel_grid.matrix.astype(np.uint8)
    out = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

    # Clip to resolution in case trimesh overshoots by 1 voxel
    d = tuple(min(s, resolution) for s in dense.shape)
    out[: d[0], : d[1], : d[2]] = dense[: d[0], : d[1], : d[2]]
    return out


def voxelize_and_save(
    obj_path: str | Path,
    voxel_out_path: str | Path,
    resolution: int = 32,
) -> np.ndarray:
    """Voxelize mesh and save binary grid; returns the array for downstream use."""
    voxel = voxelize_mesh(obj_path, resolution)
    Path(voxel_out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(voxel_out_path), voxel)
    return voxel
