"""Wrapper to run the `data.generator` module directly from the Utilities folder.

Usage:
  python run_generator.py --samples 50000 --max-depth 10 --out data/dataset

When running from the Utilities folder, this script allows direct execution of
the generator without needing the parent folder on sys.path.
"""
import sys
import os

# Ensure Utilities is on the path as a package root
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

# Import and run the generator's CLI directly
from data import generator

if __name__ == "__main__":
    # Call the generator's main block by running its __main__ logic
    import argparse
    parser = argparse.ArgumentParser(description="Generate NeuralCube F2L training data")
    parser.add_argument("--samples",   type=int, default=200_000, help="Number of samples")
    parser.add_argument("--min-depth", type=int, default=1,       help="Min scramble depth")
    parser.add_argument("--max-depth", type=int, default=12,      help="Max scramble depth")
    parser.add_argument("--bfs-limit", type=int, default=generator.BFS_DEPTH_LIMIT, help="BFS depth limit")
    parser.add_argument("--out",       type=str, default="data/dataset", help="Output file prefix")
    parser.add_argument("--seed",      type=int, default=42,      help="Random seed")
    args = parser.parse_args()

    X, y = generator.build_dataset(
        samples=args.samples,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        bfs_limit=args.bfs_limit,
        seed=args.seed,
        verbose=True,
    )
    generator.save_dataset(X, y, args.out)


