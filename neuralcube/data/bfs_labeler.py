"""
data/bfs_labeler.py
-------------------
Given a scrambled cube state, find the optimal next move using BFS.

BFS_DEPTH_LIMIT is set to 6 (down from 8).

Why 6?
  At depth 8, BFS can explore 18^8 ≈ 11 billion nodes in the worst case.
  Even with pruning and visited-set deduplication, deep states force BFS
  to build massive queues that stall for seconds per sample.
  At depth 6, BFS explores at most 18^6 ≈ 34 million nodes — still fast
  because the visited set prunes most branches. States deeper than 6
  fall through to the heuristic, which is near-instant and produces
  good-enough labels for training.

Label quality at each depth:
  depth 1–6  → BFS optimal (perfect labels)
  depth 7+   → Heuristic (good labels, not always globally optimal)

The heuristic picks the move that maximizes solved F2L slot count,
with a secondary score for cross+slot facelet correctness.
"""

from collections import deque
from ..cube.state import CubeState, MOVE_NAMES
from ..cube.f2l_checker import is_f2l_solved, count_solved_slots, is_cross_solved

# Lowered from 8 → 6 for ~100x speedup on deep states
BFS_DEPTH_LIMIT = 6


def bfs_label(cube: CubeState, depth_limit: int = BFS_DEPTH_LIMIT) -> int | None:
    """
    BFS from `cube` toward F2L-solved state.

    Returns the integer move index (0–17) of the first move on the
    shortest solution path, or None if already solved.

    Falls back to heuristic if BFS exceeds depth_limit.
    """
    if is_f2l_solved(cube):
        return None

    start = cube.facelets.tobytes()
    queue = deque()
    visited = {start}

    # Depth 1 — check immediate solutions and seed the queue
    for move_idx, move_name in enumerate(MOVE_NAMES):
        next_state = cube.apply_move(move_name)
        key = next_state.facelets.tobytes()
        if key not in visited:
            visited.add(key)
            if is_f2l_solved(next_state):
                return move_idx
            queue.append((next_state, move_idx, 1))

    # Depth 2+ BFS
    while queue:
        state, first_move_idx, depth = queue.popleft()

        if depth >= depth_limit:
            continue

        for move_name in MOVE_NAMES:
            next_state = state.apply_move(move_name)
            key = next_state.facelets.tobytes()
            if key in visited:
                continue
            visited.add(key)

            if is_f2l_solved(next_state):
                return first_move_idx

            queue.append((next_state, first_move_idx, depth + 1))

    # BFS exhausted without finding solution within depth_limit
    return _heuristic_label(cube)


def _score_state(cube: CubeState) -> float:
    """
    Composite heuristic score for a cube state.
    Higher = closer to solved F2L.

    Components:
      - slots_solved (0–4): primary driver
      - cross_intact (0/1): preserve cross (should always be 1 here)
      - correct_facelets: count of facelets in correct position (0–54)
    """
    slots = count_solved_slots(cube)
    cross = 1 if is_cross_solved(cube) else 0

    # Count facelets in correct position (fast numpy comparison)
    from ..cube.state import SOLVED_STATE
    correct = int((cube.facelets == SOLVED_STATE).sum())

    # Weighted composite: slots matter most, then facelet correctness
    return slots * 1000 + cross * 100 + correct


def _heuristic_label(cube: CubeState) -> int:
    """
    Greedy one-step lookahead: pick the move that maximises _score_state.
    Used when BFS depth limit is exceeded (depth 7+).
    """
    best_idx = 0
    best_score = -1

    for move_idx, move_name in enumerate(MOVE_NAMES):
        next_state = cube.apply_move(move_name)
        score = _score_state(next_state)
        if score > best_score:
            best_score = score
            best_idx = move_idx

    return best_idx
