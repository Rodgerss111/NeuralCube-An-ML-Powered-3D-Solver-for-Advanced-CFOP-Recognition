"""
data/generator.py
-----------------
Batch-aware dataset generator with checkpointing and resume support.

DESIGN
------
200k samples are split into 4 batches of 50k, each covering a fixed
depth range. This gives consistent speed per batch and guaranteed
coverage at all difficulty levels.

  Batch 1: depth  1–4   seed 1000   ~10 min   (easy, BFS always finds solution)
  Batch 2: depth  5–7   seed 2000   ~20 min   (medium, BFS + some heuristic)
  Batch 3: depth  8–10  seed 3000   ~30 min   (hard, mostly heuristic labels)
  Batch 4: depth 11–14  seed 4000   ~40 min   (hardest, all heuristic labels)

Each batch saves:
  data/batches/batch_<N>_X.npy
  data/batches/batch_<N>_y.npy
  data/batches/batch_<N>_meta.json   ← progress + stats

Checkpoints save every CHECKPOINT_EVERY samples so a crashed run
resumes from the last checkpoint instead of restarting.

  data/batches/batch_<N>_ckpt_X.npy
  data/batches/batch_<N>_ckpt_y.npy
  data/batches/batch_<N>_ckpt_meta.json

CLI USAGE
---------
  # Run a single batch (do this once per day)
  python -m data.generator --batch 1
  python -m data.generator --batch 2
  python -m data.generator --batch 3
  python -m data.generator --batch 4

  # Custom parameters (overrides batch defaults)
  python -m data.generator --batch 1 --samples 30000 --min-depth 1 --max-depth 4

  # Check progress across all batches
  python -m data.generator --status

  # Merge completed batches into final dataset
  python -m data.generator --merge --out data/dataset

  # Legacy: single run without batch mode
  python -m data.generator --samples 50000 --min-depth 1 --max-depth 12 --out data/run1
"""

import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm

from .scrambler import scramble
from .bfs_labeler import bfs_label, BFS_DEPTH_LIMIT
from ..cube.f2l_checker import is_f2l_solved

# ── Batch configuration ───────────────────────────────────────────────────────

BATCH_DIR = "data/batches"

BATCH_CONFIG = {
    1: {"samples": 50_000, "min_depth":  1, "max_depth":  4, "seed": 1000, "bfs_limit": 6},
    2: {"samples": 50_000, "min_depth":  5, "max_depth":  7, "seed": 2000, "bfs_limit": 0},
    3: {"samples": 50_000, "min_depth":  8, "max_depth": 10, "seed": 3000, "bfs_limit": 0},
    4: {"samples": 50_000, "min_depth": 11, "max_depth": 14, "seed": 4000, "bfs_limit": 0},
}

CHECKPOINT_EVERY = 5_000   # Save partial results every N samples


# ── Path helpers ──────────────────────────────────────────────────────────────

def _batch_prefix(batch_id: int) -> str:
    return os.path.join(BATCH_DIR, f"batch_{batch_id}")

def _ckpt_prefix(batch_id: int) -> str:
    return os.path.join(BATCH_DIR, f"batch_{batch_id}_ckpt")

def _meta_path(prefix: str) -> str:
    return f"{prefix}_meta.json"

def _x_path(prefix: str) -> str:
    return f"{prefix}_X.npy"

def _y_path(prefix: str) -> str:
    return f"{prefix}_y.npy"


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(
    batch_id: int,
    X_list: list,
    y_list: list,
    samples_attempted: int,
    skipped: int,
    elapsed: float,
    config: dict,
):
    prefix = _ckpt_prefix(batch_id)
    os.makedirs(BATCH_DIR, exist_ok=True)

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32)
    np.save(_x_path(prefix), X_arr)
    np.save(_y_path(prefix), y_arr)

    meta = {
        "batch_id":          batch_id,
        "status":            "in_progress",
        "samples_saved":     len(X_list),
        "samples_attempted": samples_attempted,
        "samples_target":    config["samples"],
        "skipped":           skipped,
        "elapsed_sec":       round(elapsed, 1),
        "config":            config,
    }
    with open(_meta_path(prefix), "w") as f:
        json.dump(meta, f, indent=2)


def _load_checkpoint(batch_id: int) -> tuple[list, list, dict] | None:
    """Load a checkpoint if it exists. Returns (X_list, y_list, meta) or None."""
    prefix = _ckpt_prefix(batch_id)
    x_path = _x_path(prefix)
    y_path = _y_path(prefix)
    meta_path = _meta_path(prefix)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None

    X = np.load(x_path)
    y = np.load(y_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    print(f"  Resuming from checkpoint: {len(X):,} samples already saved.")
    return X.tolist(), y.tolist(), meta


def _clear_checkpoint(batch_id: int):
    prefix = _ckpt_prefix(batch_id)
    for path in [_x_path(prefix), _y_path(prefix), _meta_path(prefix)]:
        if os.path.exists(path):
            os.remove(path)


# ── Core generation ───────────────────────────────────────────────────────────

def generate_batch(
    batch_id: int,
    samples: int = None,
    min_depth: int = None,
    max_depth: int = None,
    seed: int = None,
    bfs_limit: int = BFS_DEPTH_LIMIT,
    resume: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one batch of training data with checkpointing.

    Uses BATCH_CONFIG defaults unless overridden by explicit args.
    If resume=True and a checkpoint exists, continues from where it left off.

    Returns (X, y) arrays.
    """
    # Resolve config
    cfg = dict(BATCH_CONFIG.get(batch_id, BATCH_CONFIG[1]))
    if samples  is not None: cfg["samples"]   = samples
    if min_depth is not None: cfg["min_depth"] = min_depth
    if max_depth is not None: cfg["max_depth"] = max_depth
    if seed     is not None: cfg["seed"]       = seed

    target   = cfg["samples"]
    lo_depth = cfg["min_depth"]
    hi_depth = cfg["max_depth"]
    rng_seed = cfg["seed"]

    print(f"\n{'='*58}")
    print(f"  Batch {batch_id}  |  {target:,} samples  |  depth {lo_depth}–{hi_depth}  |  seed {rng_seed}")
    print(f"{'='*58}")

    # ── Resume from checkpoint ────────────────────────────────────────────
    X_list, y_list, skipped_so_far, attempted_so_far = [], [], 0, 0

    if resume:
        ckpt = _load_checkpoint(batch_id)
        if ckpt:
            X_list, y_list, meta = ckpt
            skipped_so_far   = meta.get("skipped", 0)
            attempted_so_far = meta.get("samples_attempted", 0)

    already_have = len(X_list)
    still_need   = target - already_have

    if still_need <= 0:
        print(f"  Batch {batch_id} already complete ({already_have:,} samples).")
        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32),
        )

    print(f"  Need {still_need:,} more samples (have {already_have:,}).")

    # ── Seed offset so resumed runs don't repeat states ───────────────────
    # Each checkpoint sample used one rng call, so offset seed by attempted count
    import random
    rng = random.Random(rng_seed + attempted_so_far)

    t_start = time.perf_counter()
    last_ckpt_time = t_start
    generated_this_run = 0

    bar = tqdm(total=still_need, desc=f"Batch {batch_id}", unit="samples")

    while generated_this_run < still_need:
        depth = rng.randint(lo_depth, hi_depth)
        cube, _ = scramble(depth, rng)
        attempted_so_far += 1

        if is_f2l_solved(cube):
            skipped_so_far += 1
            continue

        label = bfs_label(cube, bfs_limit)
        if label is None:
            skipped_so_far += 1
            continue

        X_list.append(cube.encode())
        y_list.append(label)
        generated_this_run += 1
        bar.update(1)

        # ── Checkpoint every N samples ────────────────────────────────────
        if generated_this_run % CHECKPOINT_EVERY == 0:
            elapsed = time.perf_counter() - t_start
            rate = generated_this_run / max(elapsed, 0.001)
            remaining = (still_need - generated_this_run) / max(rate, 0.001)

            bar.set_postfix({
                "rate": f"{rate:.0f}/s",
                "eta": f"{remaining/60:.1f}min",
                "skipped": skipped_so_far,
            })

            _save_checkpoint(
                batch_id, X_list, y_list,
                attempted_so_far, skipped_so_far,
                time.perf_counter() - t_start,
                cfg,
            )

    bar.close()

    elapsed = time.perf_counter() - t_start
    rate = generated_this_run / max(elapsed, 0.001)
    print(f"\n  Generated {generated_this_run:,} samples in {elapsed/60:.1f} min  ({rate:.0f}/s)")
    print(f"  Skipped (solved/unlabelable): {skipped_so_far:,}")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
    )


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_batch(batch_id: int, X: np.ndarray, y: np.ndarray):
    """Save a completed batch and remove its checkpoint."""
    prefix = _batch_prefix(batch_id)
    os.makedirs(BATCH_DIR, exist_ok=True)
    np.save(_x_path(prefix), X)
    np.save(_y_path(prefix), y)

    meta = {
        "batch_id":      batch_id,
        "status":        "complete",
        "samples":       len(X),
        "config":        BATCH_CONFIG.get(batch_id, {}),
        "x_shape":       list(X.shape),
        "y_shape":       list(y.shape),
    }
    with open(_meta_path(prefix), "w") as f:
        json.dump(meta, f, indent=2)

    _clear_checkpoint(batch_id)
    print(f"  Saved: {_x_path(prefix)}  {X.shape}")
    print(f"  Saved: {_y_path(prefix)}  {y.shape}")


def load_batch(batch_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a completed batch. Returns None if batch not found."""
    prefix = _batch_prefix(batch_id)
    x_path = _x_path(prefix)
    y_path = _y_path(prefix)
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"  Loaded batch {batch_id}: X={X.shape}, y={y.shape}")
    return X, y


# ── Status reporter ───────────────────────────────────────────────────────────

def print_status():
    """Print the completion status of all 4 batches."""
    print(f"\n{'='*58}")
    print("  NeuralCube Dataset Generation Status")
    print(f"{'='*58}")

    total_samples = 0
    total_target  = sum(c["samples"] for c in BATCH_CONFIG.values())

    for batch_id, cfg in BATCH_CONFIG.items():
        prefix  = _batch_prefix(batch_id)
        ckpt    = _ckpt_prefix(batch_id)
        target  = cfg["samples"]

        if os.path.exists(_x_path(prefix)):
            X = np.load(_x_path(prefix))
            n = len(X)
            pct = n / target * 100
            status = "✓ COMPLETE"
            total_samples += n
            print(f"  Batch {batch_id}  depth {cfg['min_depth']:2d}–{cfg['max_depth']:2d}  "
                  f"{n:>7,}/{target:,}  {pct:5.1f}%  {status}")
        elif os.path.exists(_x_path(ckpt)):
            X = np.load(_x_path(ckpt))
            n = len(X)
            pct = n / target * 100
            status = "⟳ IN PROGRESS"
            total_samples += n
            print(f"  Batch {batch_id}  depth {cfg['min_depth']:2d}–{cfg['max_depth']:2d}  "
                  f"{n:>7,}/{target:,}  {pct:5.1f}%  {status}")
        else:
            print(f"  Batch {batch_id}  depth {cfg['min_depth']:2d}–{cfg['max_depth']:2d}  "
                  f"{'0':>7}/{target:,}    0.0%  ✗ NOT STARTED")

    print(f"{'-'*58}")
    print(f"  Total:  {total_samples:,} / {total_target:,}  "
          f"({total_samples/total_target*100:.1f}%)")
    print(f"{'='*58}\n")


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_batches(
    out_prefix: str = "data/dataset",
    deduplicate: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge all completed batches into a single training dataset.

    Steps:
      1. Load all available completed batches.
      2. Concatenate X and y arrays.
      3. Deduplicate identical cube states (optional but recommended).
      4. Shuffle (optional but recommended before training).
      5. Save as <out_prefix>_X.npy and <out_prefix>_y.npy.

    Returns (X, y).
    """
    print(f"\nMerging batches → {out_prefix}_X/y.npy")

    X_parts, y_parts = [], []
    for batch_id in BATCH_CONFIG:
        result = load_batch(batch_id)
        if result is None:
            print(f"  WARNING: Batch {batch_id} not found — skipping.")
            continue
        X_b, y_b = result
        X_parts.append(X_b)
        y_parts.append(y_b)

    if not X_parts:
        raise RuntimeError("No completed batches found. Run batches first.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    print(f"  Combined: {len(X):,} samples")

    # ── Deduplicate ───────────────────────────────────────────────────────
    if deduplicate:
        print("  Deduplicating...")
        # Convert rows to bytes for fast hashing
        seen = set()
        keep = []
        for i in range(len(X)):
            key = X[i].tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(i)
        keep = np.array(keep)
        removed = len(X) - len(keep)
        X = X[keep]
        y = y[keep]
        print(f"  Removed {removed:,} duplicates → {len(X):,} unique samples")

    # ── Shuffle ───────────────────────────────────────────────────────────
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        print(f"  Shuffled.")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)
    print(f"  Saved: {out_prefix}_X.npy  {X.shape}")
    print(f"  Saved: {out_prefix}_y.npy  {y.shape}")

    # Save merge meta
    meta = {
        "total_samples":  int(len(X)),
        "deduplicated":   deduplicate,
        "shuffled":       shuffle,
        "batches_merged": [i for i in BATCH_CONFIG if load_batch(i) is not None],
        "x_shape":        list(X.shape),
        "y_shape":        list(y.shape),
    }
    with open(f"{out_prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return X, y


def load_dataset(out_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a previously merged/saved dataset."""
    X = np.load(f"{out_prefix}_X.npy")
    y = np.load(f"{out_prefix}_y.npy")
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    return X, y


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuralCube F2L data generator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Run batch 1 (depth 1-4, ~10 min)
  python -m data.generator --batch 1

  # Run batch 2 (depth 5-7, ~20 min)
  python -m data.generator --batch 2

  # Check all batch progress
  python -m data.generator --status

  # Merge all completed batches
  python -m data.generator --merge

  # Custom single run (legacy mode)
  python -m data.generator --samples 10000 --min-depth 1 --max-depth 6 --out data/custom
        """
    )

    # Batch mode
    parser.add_argument("--batch",     type=int, choices=[1,2,3,4],
                        help="Run a specific batch (1–4). Uses preset config.")
    parser.add_argument("--status",    action="store_true",
                        help="Show completion status of all batches.")
    parser.add_argument("--merge",     action="store_true",
                        help="Merge all completed batches into final dataset.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint and start batch from scratch.")

    # Override / legacy mode
    parser.add_argument("--samples",   type=int,   help="Override sample count")
    parser.add_argument("--min-depth", type=int,   help="Override min scramble depth")
    parser.add_argument("--max-depth", type=int,   help="Override max scramble depth")
    parser.add_argument("--seed",      type=int,   help="Override random seed")
    parser.add_argument("--bfs-limit", type=int,   default=BFS_DEPTH_LIMIT,
                        help=f"BFS depth limit (default {BFS_DEPTH_LIMIT})")
    parser.add_argument("--out",       type=str,   default="data/dataset",
                        help="Output prefix for --merge or legacy mode")

    args = parser.parse_args()

    # ── Status ────────────────────────────────────────────────────────────
    if args.status:
        print_status()

    # ── Merge ─────────────────────────────────────────────────────────────
    elif args.merge:
        merge_batches(out_prefix=args.out)

    # ── Batch mode ────────────────────────────────────────────────────────
    elif args.batch:
        X, y = generate_batch(
            batch_id  = args.batch,
            samples   = args.samples,
            min_depth = args.min_depth,
            max_depth = args.max_depth,
            seed      = args.seed,
            bfs_limit = args.bfs_limit,
            resume    = not args.no_resume,
        )
        save_batch(args.batch, X, y)
        print(f"\nBatch {args.batch} complete. Run --status to check overall progress.")

    # ── Legacy single-run mode ────────────────────────────────────────────
    elif args.samples or args.out != "data/dataset":
        import random
        rng = random.Random(args.seed or 42)
        lo  = args.min_depth or 1
        hi  = args.max_depth or 12
        n   = args.samples or 50_000

        print(f"Legacy mode: generating {n:,} samples, depth {lo}–{hi}...")

        from data.scrambler import scramble as do_scramble
        X_list, y_list, skipped = [], [], 0

        for _ in tqdm(range(n * 2), desc="Generating"):
            if len(X_list) >= n:
                break
            depth = rng.randint(lo, hi)
            cube, _ = do_scramble(depth, rng)
            if is_f2l_solved(cube):
                skipped += 1
                continue
            label = bfs_label(cube, args.bfs_limit)
            if label is None:
                skipped += 1
                continue
            X_list.append(cube.encode())
            y_list.append(label)

        X = np.array(X_list[:n], dtype=np.float32)
        y = np.array(y_list[:n], dtype=np.int32)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(f"{args.out}_X.npy", X)
        np.save(f"{args.out}_y.npy", y)
        print(f"Saved: {args.out}_X.npy  {X.shape}")
        print(f"Saved: {args.out}_y.npy  {y.shape}")
        print(f"Skipped: {skipped:,}")

    else:
        parser.print_help()
