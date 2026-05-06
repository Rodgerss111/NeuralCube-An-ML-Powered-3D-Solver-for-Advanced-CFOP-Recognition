"""
data/bfs_labeler.py
-------------------
Given a scrambled cube state, find the OPTIMAL next move using BFS.

Approach:
  - BFS from the scrambled state toward the F2L-solved state.
  - The first move in the shortest path is the label.
  - BFS is only tractable for shallow depths (≤ 10–12 moves).
  - For deeper states, we use a greedy heuristic: pick the move that
    maximizes the number of solved F2L slots (slot count heuristic).

The returned label is an integer index into MOVE_NAMES (0–17).
"""

from collections import deque
from typing import Optional, List
from cube.state import CubeState, MOVE_NAMES
from cube.f2l_checker import is_f2l_solved, count_solved_slots

# BFS depth limit — beyond this we switch to heuristic labeling
BFS_DEPTH_LIMIT = 8


def bfs_label(cube: CubeState, depth_limit: int = BFS_DEPTH_LIMIT) -> Optional[int]:
    """
    BFS from `cube` toward F2L-solved state.

    Returns the integer move index (0–17) of the first move in the
    shortest solution, or None if already solved.

    Falls back to heuristic if BFS exceeds depth_limit.
    """
    if is_f2l_solved(cube):
        return None  # Already solved — shouldn't be used as training sample

    # BFS queue: (state_bytes, first_move_index, depth)
    start = cube.facelets.tobytes()
    queue = deque()
    visited = {start}

    for move_idx, move_name in enumerate(MOVE_NAMES):
        next_state = cube.apply_move(move_name)
        key = next_state.facelets.tobytes()
        if key not in visited:
            visited.add(key)
            queue.append((next_state, move_idx, 1))
            if is_f2l_solved(next_state):
                return move_idx  # 1-move solution

    while queue:
        state, first_move_idx, depth = queue.popleft()

        if depth >= depth_limit:
            continue  # Don't expand beyond limit

        for move_name in MOVE_NAMES:
            next_state = state.apply_move(move_name)
            key = next_state.facelets.tobytes()
            if key in visited:
                continue
            visited.add(key)

            if is_f2l_solved(next_state):
                return first_move_idx  # Found solution

            queue.append((next_state, first_move_idx, depth + 1))

    # BFS exceeded depth limit — fall back to heuristic
    return _heuristic_label(cube)


def _heuristic_label(cube: CubeState) -> int:
    """
    Greedy heuristic: pick the move that maximizes F2L progress.
    Tiebreak by move index (prefer simpler moves).

    Used when BFS depth limit is exceeded.
    """
    best_move_idx = 0
    best_score = -1

    for move_idx, move_name in enumerate(MOVE_NAMES):
        next_state = cube.apply_move(move_name)
        score = count_solved_slots(next_state)
        if score > best_score:
            best_score = score
            best_move_idx = move_idx

    return best_move_idx


def batch_label(
    cubes: List[CubeState],
    depth_limit: int = BFS_DEPTH_LIMIT,
    verbose: bool = False,
) -> List[Optional[int]]:
    """
    Label a batch of cube states.
    Returns a list of move indices (int) or None for already-solved states.
    """
    labels = []
    for i, cube in enumerate(cubes):
        if verbose and i % 1000 == 0:
            print(f"  Labeling {i}/{len(cubes)}...")
        labels.append(bfs_label(cube, depth_limit))
    return labels
