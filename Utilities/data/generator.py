"""
data/generator.py
-----------------
Builds the training dataset and saves it as NumPy .npy files.

Output files:
  <out>_X.npy  — float32 array of shape (N, 324)  — encoded cube states
  <out>_y.npy  — int32   array of shape (N,)       — move label (0–17)

Usage (CLI):
  python -m data.generator --samples 200000 --max-depth 12 --out data/dataset

Usage (Python):
  from data.generator import build_dataset
  X, y = build_dataset(samples=50000, max_depth=10)
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

from .scrambler import generate_scrambles
from .bfs_labeler import bfs_label, BFS_DEPTH_LIMIT
from ..cube.f2l_checker import is_f2l_solved


def build_dataset(
    samples: int = 200_000,
    min_depth: int = 1,
    max_depth: int = 12,
    bfs_limit: int = BFS_DEPTH_LIMIT,
    curriculum: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a (X, y) dataset of encoded cube states and optimal move labels.

    Parameters
    ----------
    samples    : total number of training samples to generate
    min_depth  : minimum scramble depth
    max_depth  : maximum scramble depth
    bfs_limit  : BFS depth before falling back to heuristic label
    curriculum : if True, bias toward shallow scrambles early in dataset
    seed       : random seed for reproducibility
    verbose    : show progress bar

    Returns
    -------
    X : np.ndarray of shape (N, 324), dtype float32
    y : np.ndarray of shape (N,),     dtype int32
    """
    if verbose:
        print(f"Generating {samples:,} samples (depth {min_depth}–{max_depth})...")

    scrambles = generate_scrambles(
        n=samples,
        min_depth=min_depth,
        max_depth=max_depth,
        curriculum=curriculum,
        seed=seed,
    )

    X_list = []
    y_list = []
    skipped = 0

    iterator = tqdm(scrambles, desc="Labeling", disable=not verbose)
    for cube, _, depth in iterator:
        # Skip already-solved states
        if is_f2l_solved(cube):
            skipped += 1
            continue

        label = bfs_label(cube, bfs_limit)
        if label is None:
            skipped += 1
            continue

        X_list.append(cube.encode())
        y_list.append(label)

    if verbose:
        print(f"  Done. Generated {len(X_list):,} samples, skipped {skipped:,}.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def save_dataset(X: np.ndarray, y: np.ndarray, out_prefix: str):
    """Save X and y arrays as <out_prefix>_X.npy and <out_prefix>_y.npy."""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)
    print(f"Saved: {out_prefix}_X.npy  ({X.shape})")
    print(f"Saved: {out_prefix}_y.npy  ({y.shape})")


def load_dataset(out_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load previously saved dataset."""
    X = np.load(f"{out_prefix}_X.npy")
    y = np.load(f"{out_prefix}_y.npy")
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NeuralCube F2L training data")
    parser.add_argument("--samples",   type=int, default=200_000, help="Number of samples")
    parser.add_argument("--min-depth", type=int, default=1,       help="Min scramble depth")
    parser.add_argument("--max-depth", type=int, default=12,      help="Max scramble depth")
    parser.add_argument("--bfs-limit", type=int, default=BFS_DEPTH_LIMIT, help="BFS depth limit")
    parser.add_argument("--out",       type=str, default="data/dataset", help="Output file prefix")
    parser.add_argument("--seed",      type=int, default=42,      help="Random seed")
    args = parser.parse_args()

    X, y = build_dataset(
        samples=args.samples,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        bfs_limit=args.bfs_limit,
        seed=args.seed,
        verbose=True,
    )
    save_dataset(X, y, args.out)
