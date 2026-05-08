"""
data/bfs_labeler.py
-------------------
Optimal-move labeler using BFS for shallow states and a
2-ply lookahead heuristic for deep states.
 
LABELING STRATEGY
-----------------
depth 1–6  → BFS (optimal labels, fast at these depths)
depth 7+   → 2-ply heuristic (near-instant, near-optimal quality)
 
WHY 2-PLY?
----------
The old 1-ply heuristic scored each of the 18 candidate moves and
picked the best — like a chess player who only thinks one move ahead.
The problem: some moves look bad immediately but open up a much better
position on the very next move (e.g. a setup move before inserting an
F2L pair). A 1-ply lookahead misses these entirely and labels them as
wrong moves.
 
2-ply fixes this by scoring each (move1, move2) pair — 18 × 18 = 324
`apply_move` calls total — and returning the move1 that leads to the
best reachable state in 2 steps. This is analogous to a chess player
who thinks "if I play this move, what's the best my opponent can do,
and how good is my resulting position?" — except here move2 is also
ours (we're solving alone), so we pick the best 2-move sequence and
label with its first move.
 
Cost comparison:
  1-ply :  18 apply_move calls  — very fast, weak labels
  2-ply : 324 apply_move calls  — still near-instant, much better labels
  BFS-6 :  up to ~34M visits   — slow for deep states, optimal labels
"""
 
from collections import deque
from cube.state import CubeState, MOVE_NAMES, SOLVED_STATE
from cube.f2l_checker import is_f2l_solved, count_solved_slots, is_cross_solved
 
# BFS depth limit — states deeper than this use 2-ply heuristic
BFS_DEPTH_LIMIT = 6
 
 
# ── Scoring function ──────────────────────────────────────────────────────────
 
def _score_state(cube: CubeState) -> float:
    """
    Composite progress score for a cube state. Higher = closer to solved F2L.
 
    Components (weighted):
      slots_solved   (0–4)  × 1000  — primary: how many F2L slots are done
      cross_intact   (0/1)  × 500   — penalise hard if cross is broken
      correct_facelets (0–54) × 1   — fine-grained tiebreaker
 
    Weights are chosen so that gaining one slot always outscores any
    improvement in the tiebreakers, and cross preservation outscores
    facelet count.
    """
    slots   = count_solved_slots(cube)
    cross   = 1 if is_cross_solved(cube) else 0
    correct = int((cube.facelets == SOLVED_STATE).sum())
    return slots * 1000 + cross * 500 + correct
 
 
# ── BFS labeler ───────────────────────────────────────────────────────────────
 
def bfs_label(cube: CubeState, depth_limit: int = BFS_DEPTH_LIMIT) -> int | None:
    """
    Find the optimal next move using BFS up to depth_limit.
    Falls back to 2-ply heuristic if depth_limit is exceeded or set to 0.
 
    Returns move index (0–17) or None if already solved.
    """
    if is_f2l_solved(cube):
        return None
 
    # Skip BFS entirely if limit is 0 (batch 2–4 fast mode)
    if depth_limit == 0:
        return _heuristic_2ply(cube)
 
    start = cube.facelets.tobytes()
    queue = deque()
    visited = {start}
 
    # Depth 1
    for move_idx, move_name in enumerate(MOVE_NAMES):
        next_state = cube.apply_move(move_name)
        key = next_state.facelets.tobytes()
        if key not in visited:
            visited.add(key)
            if is_f2l_solved(next_state):
                return move_idx
            queue.append((next_state, move_idx, 1))
 
    # Depth 2+
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
 
    # BFS exhausted — fall back to 2-ply heuristic
    return _heuristic_2ply(cube)
 
 
# ── 2-ply heuristic ───────────────────────────────────────────────────────────
 
def _heuristic_2ply(cube: CubeState) -> int:
    """
    2-ply lookahead heuristic: evaluate all (move1, move2) pairs (18×18=324
    apply_move calls) and return the move1 that leads to the best 2-step
    outcome.
 
    Analogy: instead of picking the door that looks best right now, you
    peek behind each door AND the next door behind that, then pick the
    first door whose hallway leads to the best room.
 
    This correctly handles setup moves — moves that look neutral or even
    slightly worse at ply-1 but unlock a much better position at ply-2
    (e.g. moving an edge out of the way before inserting a corner-edge pair).
    """
    best_move1_idx = 0
    best_score = -1
 
    for move1_idx, move1_name in enumerate(MOVE_NAMES):
        state1 = cube.apply_move(move1_name)
 
        # Early exit: if move1 already solves F2L, it's optimal
        if is_f2l_solved(state1):
            return move1_idx
 
        # Look one step further: what's the best we can reach in move2?
        best_ply2_score = _score_state(state1)  # baseline: stop after move1
 
        for move2_name in MOVE_NAMES:
            state2 = state1.apply_move(move2_name)
 
            if is_f2l_solved(state2):
                # move1 leads to a 2-move solution — very strong
                best_ply2_score = float("inf")
                break
 
            s = _score_state(state2)
            if s > best_ply2_score:
                best_ply2_score = s
 
        # move1 is judged by the best position reachable after move2
        if best_ply2_score > best_score:
            best_score = best_ply2_score
            best_move1_idx = move1_idx
 
    return best_move1_idx