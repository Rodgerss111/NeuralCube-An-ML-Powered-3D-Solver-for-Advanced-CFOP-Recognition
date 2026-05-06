"""
eval/benchmark.py
-----------------
Evaluates the trained F2L solver model.

Metrics:
  1. Solve Rate     — % of test states fully solved within move limit
  2. Move Efficiency— avg(optimal_moves / model_moves) for solved states
  3. Avg Move Count — average moves taken by the model

Usage:
  python -m eval.benchmark --model model/saved/f2l_model.h5 --samples 5000

The benchmark also shows a breakdown by scramble depth (1–5, 6–10, 11–15).
"""

import argparse
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple, List
from tqdm import tqdm

from cube.state import CubeState, MOVE_NAMES
from cube.f2l_checker import is_f2l_solved
from data.scrambler import scramble
from data.bfs_labeler import bfs_label


MAX_SOLVE_MOVES = 30   # Hard cap on inference loop iterations


def greedy_solve(model, cube: CubeState, max_moves: int = MAX_SOLVE_MOVES) -> Tuple[List[str], bool]:
    """
    Solve F2L greedily: at each step pick the highest-probability move.

    Returns (move_sequence, solved_flag).
    """
    moves_taken = []
    current = cube

    for _ in range(max_moves):
        if is_f2l_solved(current):
            return moves_taken, True

        encoded = current.encode().reshape(1, -1)
        probs = model.predict(encoded, verbose=0)[0]
        move_idx = int(np.argmax(probs))
        move_name = MOVE_NAMES[move_idx]

        current = current.apply_move(move_name)
        moves_taken.append(move_name)

    return moves_taken, is_f2l_solved(current)


def optimal_length(cube: CubeState, limit: int = 10) -> Optional[int]:
    """BFS to find optimal solution length. Returns None if > limit."""
    if is_f2l_solved(cube):
        return 0

    from collections import deque
    visited = {cube.facelets.tobytes()}
    queue = deque([(cube, 0)])

    while queue:
        state, depth = queue.popleft()
        if depth >= limit:
            continue
        for move in MOVE_NAMES:
            ns = state.apply_move(move)
            key = ns.facelets.tobytes()
            if key in visited:
                continue
            visited.add(key)
            if is_f2l_solved(ns):
                return depth + 1
            queue.append((ns, depth + 1))

    return None  # Beyond BFS limit


def run_benchmark(
    model,
    n_samples: int = 2000,
    min_depth: int = 1,
    max_depth: int = 12,
    max_moves: int = MAX_SOLVE_MOVES,
    seed: int = 99,
) -> dict:
    """
    Run the full benchmark and return a results dict.
    """
    import random
    rng = random.Random(seed)

    solved_count = 0
    total_model_moves = []
    efficiency_ratios = []

    depth_buckets = defaultdict(lambda: {"total": 0, "solved": 0})

    print(f"Running benchmark on {n_samples} test states (depth {min_depth}–{max_depth})...")

    for _ in tqdm(range(n_samples)):
        depth = rng.randint(min_depth, max_depth)
        cube, _ = scramble(depth, rng)

        if is_f2l_solved(cube):
            continue

        moves_taken, solved = greedy_solve(model, cube, max_moves)

        bucket = f"{((depth-1)//5)*5+1}-{((depth-1)//5)*5+5}"
        depth_buckets[bucket]["total"] += 1

        if solved:
            solved_count += 1
            depth_buckets[bucket]["solved"] += 1
            total_model_moves.append(len(moves_taken))

            opt = optimal_length(cube, limit=10)
            if opt is not None and len(moves_taken) > 0:
                efficiency_ratios.append(opt / len(moves_taken))

    total_tested = sum(b["total"] for b in depth_buckets.values())
    solve_rate = solved_count / max(total_tested, 1)
    avg_moves = np.mean(total_model_moves) if total_model_moves else float("nan")
    avg_efficiency = np.mean(efficiency_ratios) if efficiency_ratios else float("nan")

    return {
        "total_tested": total_tested,
        "solved": solved_count,
        "solve_rate": solve_rate,
        "avg_model_moves": avg_moves,
        "avg_efficiency_ratio": avg_efficiency,
        "depth_buckets": dict(depth_buckets),
    }


def print_report(results: dict):
    print("\n" + "="*55)
    print("  NeuralCube F2L Benchmark Report")
    print("="*55)
    print(f"  Tested        : {results['total_tested']:,}")
    print(f"  Solved        : {results['solved']:,}")
    print(f"  Solve Rate    : {results['solve_rate']*100:.1f}%")
    print(f"  Avg Moves     : {results['avg_model_moves']:.2f}")
    print(f"  Efficiency    : {results['avg_efficiency_ratio']:.3f}  (optimal/model, higher=better)")
    print("-"*55)
    print("  Breakdown by scramble depth:")
    for bucket, data in sorted(results["depth_buckets"].items()):
        rate = data["solved"] / max(data["total"], 1) * 100
        print(f"    Depth {bucket:5s}: {data['solved']}/{data['total']}  ({rate:.0f}%)")
    print("="*55)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NeuralCube F2L solver")
    parser.add_argument("--model",     type=str, required=True, help="Path to .h5 model")
    parser.add_argument("--samples",   type=int, default=2000,  help="Number of test states")
    parser.add_argument("--max-depth", type=int, default=12,    help="Max scramble depth")
    parser.add_argument("--max-moves", type=int, default=30,    help="Max moves per solve attempt")
    parser.add_argument("--seed",      type=int, default=99,    help="Random seed")
    args = parser.parse_args()

    from model.network import load_model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    results = run_benchmark(
        model,
        n_samples=args.samples,
        max_depth=args.max_depth,
        max_moves=args.max_moves,
        seed=args.seed,
    )
    print_report(results)
