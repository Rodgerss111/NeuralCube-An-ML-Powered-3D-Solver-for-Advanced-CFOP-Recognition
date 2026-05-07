"""
eval/benchmark.py
-----------------
Benchmarks all three solver phases independently and as a combined pipeline.

Reports per phase:
  - Solve rate (%)
  - Avg move count
  - Avg time (ms)
  - Breakdown by scramble depth bucket

Usage:
  # Full pipeline benchmark (requires trained model)
  python -m eval.benchmark --model model/saved/f2l_model.h5 --samples 2000

  # Rule-based only (no model needed)
  python -m eval.benchmark --rules-only --samples 2000

  # Quick smoke test
  python -m eval.benchmark --rules-only --samples 100 --max-depth 8
"""

import argparse
import time
import random
from collections import defaultdict
from tqdm import tqdm

from cube.state import CubeState
from cube.f2l_checker import is_f2l_solved
from data.scrambler import scramble
from solver.f2l_case_solver import solve_f2l
from solver.pipeline import solve


# ─────────────────────────────────────────────────────────────────────────────

def _bucket(depth: int) -> str:
    lo = ((depth - 1) // 5) * 5 + 1
    hi = lo + 4
    return f"{lo:2d}-{hi:2d}"


def benchmark_rules(
    n_samples: int = 2000,
    min_depth: int = 1,
    max_depth: int = 12,
    seed: int = 99,
) -> dict:
    """Benchmark the rule-based 41-case solver in isolation."""
    rng = random.Random(seed)
    solved_count = 0
    total_moves = []
    total_times = []
    buckets = defaultdict(lambda: {"total": 0, "solved": 0, "moves": []})

    for _ in tqdm(range(n_samples), desc="Rules benchmark"):
        depth = rng.randint(min_depth, max_depth)
        cube, _ = scramble(depth, rng)
        if is_f2l_solved(cube):
            continue

        t0 = time.perf_counter()
        moves, solved = solve_f2l(cube)
        elapsed = (time.perf_counter() - t0) * 1000

        b = _bucket(depth)
        buckets[b]["total"] += 1
        total_times.append(elapsed)

        if solved:
            solved_count += 1
            buckets[b]["solved"] += 1
            buckets[b]["moves"].append(len(moves))
            total_moves.append(len(moves))

    total = sum(b["total"] for b in buckets.values())
    return {
        "solver": "rule_based_41_cases",
        "total": total,
        "solved": solved_count,
        "solve_rate": solved_count / max(total, 1),
        "avg_moves": sum(total_moves) / max(len(total_moves), 1),
        "avg_time_ms": sum(total_times) / max(len(total_times), 1),
        "buckets": dict(buckets),
    }


def benchmark_pipeline(
    model,
    n_samples: int = 2000,
    min_depth: int = 1,
    max_depth: int = 12,
    seed: int = 99,
) -> dict:
    """Benchmark the full cascading pipeline (NN → rules → Kociemba)."""
    rng = random.Random(seed)
    solved_count = 0
    total_moves = []
    total_times = []
    phase_counts = defaultdict(int)
    buckets = defaultdict(lambda: {"total": 0, "solved": 0})

    for _ in tqdm(range(n_samples), desc="Pipeline benchmark"):
        depth = rng.randint(min_depth, max_depth)
        cube, _ = scramble(depth, rng)
        if is_f2l_solved(cube):
            continue

        result = solve(
            cube, model=model,
            use_nn=(model is not None),
            use_case_solver=True,
            use_kociemba=True,
        )

        b = _bucket(depth)
        buckets[b]["total"] += 1
        total_times.append(result["time_ms"])
        phase_counts[result["phase_used"]] += 1

        if result["solved"]:
            solved_count += 1
            buckets[b]["solved"] += 1
            total_moves.append(result["move_count"])

    total = sum(b["total"] for b in buckets.values())
    return {
        "solver": "full_pipeline",
        "total": total,
        "solved": solved_count,
        "solve_rate": solved_count / max(total, 1),
        "avg_moves": sum(total_moves) / max(len(total_moves), 1),
        "avg_time_ms": sum(total_times) / max(len(total_times), 1),
        "phase_breakdown": dict(phase_counts),
        "buckets": dict(buckets),
    }


def print_report(results: dict):
    name = results["solver"].upper().replace("_", " ")
    print("\n" + "=" * 58)
    print(f"  {name}")
    print("=" * 58)
    print(f"  Tested     : {results['total']:,}")
    print(f"  Solved     : {results['solved']:,}")
    print(f"  Solve Rate : {results['solve_rate']*100:.1f}%")
    print(f"  Avg Moves  : {results['avg_moves']:.1f}")
    print(f"  Avg Time   : {results['avg_time_ms']:.1f} ms")
    if "phase_breakdown" in results:
        print(f"  Phase used :")
        for phase, count in sorted(results["phase_breakdown"].items()):
            print(f"    {phase:15s}: {count:,}")
    print("-" * 58)
    print("  Depth  Solved / Tested  Rate")
    for bucket in sorted(results["buckets"]):
        b = results["buckets"][bucket]
        rate = b["solved"] / max(b["total"], 1) * 100
        print(f"  {bucket}    {b['solved']:5d} / {b['total']:5d}    {rate:5.1f}%")
    print("=" * 58)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralCube F2L benchmark")
    parser.add_argument("--model",      type=str,  default=None,  help="Path to .h5 model")
    parser.add_argument("--samples",    type=int,  default=2000,  help="Test samples")
    parser.add_argument("--min-depth",  type=int,  default=1)
    parser.add_argument("--max-depth",  type=int,  default=12)
    parser.add_argument("--seed",       type=int,  default=99)
    parser.add_argument("--rules-only", action="store_true", help="Benchmark rules solver only")
    args = parser.parse_args()

    if args.rules_only or args.model is None:
        print("Benchmarking rule-based solver...")
        r = benchmark_rules(args.samples, args.min_depth, args.max_depth, args.seed)
        print_report(r)
    else:
        from model.network import load_model
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)

        print("\nBenchmarking full pipeline...")
        r = benchmark_pipeline(model, args.samples, args.min_depth, args.max_depth, args.seed)
        print_report(r)

        print("\nBenchmarking rule-based solver alone for comparison...")
        r2 = benchmark_rules(args.samples, args.min_depth, args.max_depth, args.seed)
        print_report(r2)
