"""
data/scrambler.py
-----------------
Generates scrambled cube states starting from a solved F2L state.

Strategy:
  - Apply N random moves (from the 18 HTM moves) to a solved cube.
  - Optionally filter out states where F2L is already solved (useless training samples).
  - Supports curriculum learning by letting the caller control scramble depth.

Note: We deliberately avoid using move-inverse pairs back-to-back (e.g. R then R')
since they cancel out and produce shallower-than-expected scrambles.
"""

import random
import numpy as np
from ..cube.state import CubeState, MOVE_NAMES


# Moves grouped by face — used to avoid consecutive same-face moves
_FACE_OF = {m: m.rstrip("'2") for m in MOVE_NAMES}

# Opposite faces — avoid redundant commuting moves
_OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def scramble(depth: int, rng: random.Random = None) -> tuple[CubeState, list[str]]:
    """
    Apply `depth` random moves to a solved cube and return
    (scrambled_state, list_of_moves_applied).

    Avoids:
      - Two consecutive moves on the same face
      - Move immediately followed by its inverse or double on same face
    """
    if rng is None:
        rng = random.Random()

    cube = CubeState()
    moves_applied = []
    last_face = None
    second_last_face = None

    available = list(MOVE_NAMES)

    for _ in range(depth):
        # Filter: no same-face repeat; no opposite-face repeat if 2 in a row
        def allowed(m):
            face = _FACE_OF[m]
            if face == last_face:
                return False
            if face == _OPPOSITE.get(last_face) and face == second_last_face:
                return False
            return True

        candidates = [m for m in available if allowed(m)]
        if not candidates:
            candidates = available  # fallback (shouldn't happen)

        move = rng.choice(candidates)
        cube = cube.apply_move(move)
        moves_applied.append(move)
        second_last_face = last_face
        last_face = _FACE_OF[move]

    return cube, moves_applied


def generate_scrambles(
    n: int,
    min_depth: int = 1,
    max_depth: int = 12,
    curriculum: bool = True,
    seed: int = None,
) -> list[tuple[CubeState, list[str], int]]:
    """
    Generate `n` scrambled states.

    Returns list of (cube_state, moves_applied, depth) tuples.

    If curriculum=True, depths are sampled with bias toward shallower states
    early in the list, gradually increasing — simulates curriculum learning.
    """
    rng = random.Random(seed)
    results = []

    for i in range(n):
        if curriculum:
            # Progress 0→1 through the dataset
            progress = i / max(n - 1, 1)
            # Beta distribution skewed early → shallow, late → deep
            lo = min_depth + int(progress * (max_depth - min_depth) * 0.5)
            hi = min_depth + int(progress * (max_depth - min_depth)) + 1
            hi = min(hi, max_depth)
            lo = min(lo, hi)
            depth = rng.randint(lo, hi)
        else:
            depth = rng.randint(min_depth, max_depth)

        cube, moves = scramble(depth, rng)
        results.append((cube, moves, depth))

    return results
