"""Data preparation script: voxelize all raw .obj meshes and compute TSDFs.

Walks data/raw/ for .obj files, saves:
  - data/processed/voxel_{res}/<shape_id>.npy  — (res,res,res) uint8 binary
  - data/processed/tsdf_{res}/<shape_id>.npy   — (res,res,res) float32 in [-1,1]

Then generates 80/10/10 train/val/test splits in data/splits/.

Usage:
    python scripts/prepare_data.py --data_root data/ --resolution 64 --truncation 3.0 --workers 16
"""

import argparse
import multiprocessing as mp
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.voxelize import voxelize_and_save
from preprocessing.tsdf import compute_and_save_tsdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ShapeNet meshes.")
    parser.add_argument("--data_root",  type=str,   default="data/")
    parser.add_argument("--resolution", type=int,   default=64)
    parser.add_argument("--truncation", type=float, default=3.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--workers",    type=int,   default=16,
                        help="Number of parallel CPU workers.")
    parser.add_argument("--timeout",    type=int,   default=120,
                        help="Max seconds per mesh before skipping it.")
    return parser.parse_args()


def write_split(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def _process_one(args: tuple) -> tuple[str, str | None]:
    """Worker: voxelize one mesh and compute its TSDF.

    Returns (shape_id, None) on success or (shape_id, error_str) on failure.
    Skips silently if output already exists.
    """
    obj_path, voxel_path, tsdf_path, resolution, truncation, shape_id = args

    if Path(tsdf_path).exists():
        return (shape_id, None)

    try:
        voxel = voxelize_and_save(obj_path, voxel_path, resolution)
        compute_and_save_tsdf(voxel, tsdf_path, truncation)
        return (shape_id, None)
    except Exception as exc:
        return (shape_id, str(exc))


def main() -> None:
    args = parse_args()
    data_root  = Path(args.data_root)
    raw_dir    = data_root / "raw"
    res        = args.resolution
    voxel_dir  = data_root / "processed" / f"voxel_{res}"
    tsdf_dir   = data_root / "processed" / f"tsdf_{res}"
    splits_dir = data_root / "splits"

    voxel_dir.mkdir(parents=True, exist_ok=True)
    tsdf_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(raw_dir.rglob("*.obj"))
    if not obj_files:
        print(f"No .obj files found under {raw_dir}.")
        return

    print(f"Found {len(obj_files)} meshes — {res}³, {args.workers} workers, {args.timeout}s timeout")

    tasks = []
    for obj_path in obj_files:
        shape_id   = str(obj_path.relative_to(raw_dir).with_suffix("")).replace("/", "__")
        voxel_path = str(voxel_dir / f"{shape_id}.npy")
        tsdf_path  = str(tsdf_dir  / f"{shape_id}.npy")
        tasks.append((str(obj_path), voxel_path, tsdf_path, res, args.truncation, shape_id))

    shape_ids: list[str] = []
    failed:    list[tuple[str, str]] = []

    # maxtasksperchild restarts each worker every N tasks — limits memory buildup
    # and contains crashes to one worker rather than the whole pool
    with mp.Pool(processes=args.workers, maxtasksperchild=20) as pool:
        async_results = [
            (task[5], pool.apply_async(_process_one, (task,)))
            for task in tasks
        ]

        with tqdm(total=len(tasks), desc=f"Preprocessing {res}³") as pbar:
            for shape_id, result in async_results:
                try:
                    sid, err = result.get(timeout=args.timeout)
                    if err is None:
                        shape_ids.append(sid)
                    else:
                        failed.append((sid, err))
                except mp.TimeoutError:
                    failed.append((shape_id, f"timeout >{args.timeout}s"))
                except Exception as exc:
                    failed.append((shape_id, str(exc)))
                pbar.update(1)

    print(f"\nSucceeded: {len(shape_ids)}  Failed: {len(failed)}")

    if failed:
        print(f"\nFailed meshes ({len(failed)}):")
        for sid, err in failed[:10]:
            print(f"  {sid}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

        # Save failed IDs so you can inspect them
        fail_log = data_root / "failed_meshes.txt"
        fail_log.write_text("\n".join(f"{s}\t{e}" for s, e in failed) + "\n")
        print(f"\nFull failure list saved to {fail_log}")

    if not shape_ids:
        print("No shapes successfully processed.")
        return

    # 80/10/10 split
    random.seed(args.seed)
    random.shuffle(shape_ids)
    n       = len(shape_ids)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_ids = shape_ids[:n_train]
    val_ids   = shape_ids[n_train : n_train + n_val]
    test_ids  = shape_ids[n_train + n_val :]

    write_split(splits_dir / "train.txt", train_ids)
    write_split(splits_dir / "val.txt",   val_ids)
    write_split(splits_dir / "test.txt",  test_ids)

    print(f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
    print(f"Voxel dir : {voxel_dir}/")
    print(f"TSDF dir  : {tsdf_dir}/")
    print(f"Splits    : {splits_dir}/")


if __name__ == "__main__":
    main()
