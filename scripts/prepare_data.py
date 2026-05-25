"""Data preparation script: voxelize all raw .obj meshes and compute TSDFs.

Walks data/raw/ for .obj files, saves:
  - data/processed/voxel/<shape_id>.npy  — (32,32,32) uint8 binary
  - data/processed/tsdf/<shape_id>.npy   — (32,32,32) float32 in [-1,1]

Then generates 80/10/10 train/val/test splits in data/splits/.

Usage:
    python scripts/prepare_data.py --data_root data/ --resolution 32 --truncation 3.0
"""

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.voxelize import voxelize_and_save
from preprocessing.tsdf import compute_and_save_tsdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ShapeNet meshes.")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--truncation", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_split(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_dir = data_root / "raw"
    voxel_dir = data_root / "processed" / "voxel"
    tsdf_dir = data_root / "processed" / "tsdf"
    splits_dir = data_root / "splits"

    voxel_dir.mkdir(parents=True, exist_ok=True)
    tsdf_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(raw_dir.rglob("*.obj"))
    if not obj_files:
        print(f"No .obj files found under {raw_dir}. Place ShapeNet meshes there first.")
        return

    print(f"Found {len(obj_files)} meshes. Processing...")
    shape_ids: list[str] = []

    for obj_path in tqdm(obj_files, desc="Preprocessing"):
        # Use relative path from raw/ as the shape ID (with / replaced by __)
        shape_id = str(obj_path.relative_to(raw_dir).with_suffix("")).replace("/", "__")
        shape_ids.append(shape_id)

        voxel_path = voxel_dir / f"{shape_id}.npy"
        tsdf_path = tsdf_dir / f"{shape_id}.npy"

        try:
            voxel = voxelize_and_save(obj_path, voxel_path, args.resolution)
            compute_and_save_tsdf(voxel, tsdf_path, args.truncation)
        except Exception as exc:
            print(f"\n  WARNING: failed to process {obj_path}: {exc}")
            shape_ids.pop()

    if not shape_ids:
        print("No shapes successfully processed.")
        return

    # 80/10/10 split
    random.seed(args.seed)
    random.shuffle(shape_ids)
    n = len(shape_ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_ids = shape_ids[:n_train]
    val_ids = shape_ids[n_train : n_train + n_val]
    test_ids = shape_ids[n_train + n_val :]

    write_split(splits_dir / "train.txt", train_ids)
    write_split(splits_dir / "val.txt", val_ids)
    write_split(splits_dir / "test.txt", test_ids)

    print(f"\nDone. {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
    print(f"Splits written to {splits_dir}/")


if __name__ == "__main__":
    main()
