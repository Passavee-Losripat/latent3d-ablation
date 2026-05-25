"""ShapeNet dataset loader for voxel-based 3D representations.

Loads pre-processed .npy files from data/processed/{representation}/ and
returns tensors of shape (1, 32, 32, 32) along with their shape IDs.

Tensor ranges:
  - tsdf:      float32 in [-1.0, 1.0]
  - voxel:     float32 in {0.0, 1.0}
  - occupancy: float32 in {0.0, 1.0}  (same layout as voxel, added by other variants)
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


_AUGMENT_AXES = [(0, 1), (0, 2), (1, 2)]  # pairs of spatial axes for 90° rotations


class ShapeNetDataset(Dataset):
    """Dataset for a single representation type stored as .npy files."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        representation: str = "tsdf",
        augment: bool = False,
    ) -> None:
        """
        Args:
            data_root:       root data directory (contains processed/ and splits/).
            split:           'train', 'val', or 'test'.
            representation:  subdirectory name under processed/ (e.g. 'tsdf', 'voxel').
            augment:         if True, apply random 90° rotations at load time.
        """
        data_root = Path(data_root)
        splits_dir = data_root / "splits"
        self.processed_dir = data_root / "processed" / representation
        self.augment = augment

        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.shape_ids: list[str] = [
            line.strip()
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

        # Filter to shapes that actually have processed files
        self.shape_ids = [
            sid for sid in self.shape_ids
            if (self.processed_dir / f"{sid}.npy").exists()
        ]

        if not self.shape_ids:
            raise RuntimeError(
                f"No processed .npy files found in {self.processed_dir} for split '{split}'."
            )

    def __len__(self) -> int:
        return len(self.shape_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Return (tensor of shape (1, 32, 32, 32), shape_id)."""
        shape_id = self.shape_ids[idx]
        arr = np.load(str(self.processed_dir / f"{shape_id}.npy"))
        arr = arr.astype(np.float32)

        if self.augment:
            arr = _random_rotate90(arr)

        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 32, 32, 32)
        return tensor, shape_id


def _random_rotate90(arr: np.ndarray) -> np.ndarray:
    """Apply a random 90° rotation in one of the three spatial planes."""
    axes = random.choice(_AUGMENT_AXES)
    k = random.randint(1, 3)  # number of 90° turns
    return np.rot90(arr, k=k, axes=axes).copy()
