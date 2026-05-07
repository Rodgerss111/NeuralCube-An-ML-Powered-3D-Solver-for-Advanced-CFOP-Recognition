"""
solver/nn_solver.py
--------------------
Neural network inference engine for F2L solving.

Improvements over the basic greedy loop:
  1. Cross-guard  — skips any move that would disturb the solved cross.
  2. Loop detector — tracks visited states; triggers fallback if loop detected.
  3. Beam search  — optional width-k beam for better solution quality.
  4. State rollback — returns ORIGINAL state if cross is accidentally broken.
"""

import numpy as np
from cube.state import CubeState, MOVE_NAMES
from cube.f2l_checker import is_f2l_solved, is_cross_solved

# Cross facelets that must not change (from f2l_checker.py)
CROSS_INDICES = [49, 46, 25, 50, 34, 52, 43, 48, 16]

MAX_MOVES = 30
BEAM_WIDTH = 3   # set to 1 for pure greedy


# ─────────────────────────────────────────────────────────────────────────────
# Cross guard
# ─────────────────────────────────────────────────────────────────────────────

def _breaks_cross(cube: CubeState, move: str) -> bool:
    """
    Return True if applying `move` to `cube` would disturb any cross facelet.
    """
    new_state = cube.apply_move(move)
    for idx in CROSS_INDICES:
        if new_state.facelets[idx] != cube.facelets[idx]:
            # Cross facelet changed — only acceptable if it stays the correct color
            from cube.state import SOLVED_STATE
            if new_state.facelets[idx] != SOLVED_STATE[idx]:
                return True
    return False


def _safe_moves(cube: CubeState, probs: np.ndarray) -> list[tuple[float, str]]:
    """
    Return (prob, move_name) pairs sorted descending,
    filtering out moves that would break the cross.
    """
    ranked = sorted(
        [(probs[i], MOVE_NAMES[i]) for i in range(len(MOVE_NAMES))],
        reverse=True
    )
    return [(p, m) for p, m in ranked if not _breaks_cross(cube, m)]


# ─────────────────────────────────────────────────────────────────────────────
# Greedy solver (beam width = 1)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_solve(
    model,
    cube: CubeState,
    max_moves: int = MAX_MOVES,
) -> tuple[list[str], bool, str]:
    """
    Solve F2L greedily with cross-guard and loop detection.

    Returns:
        moves_taken : list of move strings applied
        solved      : True if F2L is fully solved
        exit_reason : 'solved' | 'loop_detected' | 'move_cap' | 'no_safe_moves'
    """
    moves_taken = []
    current = cube.copy()
    visited = {current.facelets.tobytes()}

    for step in range(max_moves):
        if is_f2l_solved(current):
            return moves_taken, True, "solved"

        encoded = current.encode().reshape(1, -1)
        probs = model.predict(encoded, verbose=0)[0]

        safe = _safe_moves(current, probs)
        if not safe:
            return moves_taken, False, "no_safe_moves"

        _, best_move = safe[0]
        next_state = current.apply_move(best_move)
        key = next_state.facelets.tobytes()

        # Loop detection
        if key in visited:
            return moves_taken, False, "loop_detected"

        visited.add(key)
        current = next_state
        moves_taken.append(best_move)

    solved = is_f2l_solved(current)
    return moves_taken, solved, "solved" if solved else "move_cap"


# ─────────────────────────────────────────────────────────────────────────────
# Beam search solver
# ─────────────────────────────────────────────────────────────────────────────

def beam_solve(
    model,
    cube: CubeState,
    max_moves: int = MAX_MOVES,
    beam_width: int = BEAM_WIDTH,
) -> tuple[list[str], bool, str]:
    """
    Beam search solver — keeps top-k candidate paths at each step.
    Produces better solutions than greedy at the cost of more model calls.

    Each beam entry: (cumulative_log_prob, cube_state, moves_so_far, visited_set)
    """
    import math

    if is_f2l_solved(cube):
        return [], True, "solved"

    # Initialize beam
    # (score, state, moves, visited)
    init_visited = {cube.facelets.tobytes()}
    beams = [(0.0, cube.copy(), [], init_visited)]

    for step in range(max_moves):
        candidates = []

        for score, state, moves, visited in beams:
            if is_f2l_solved(state):
                return moves, True, "solved"

            encoded = state.encode().reshape(1, -1)
            probs = model.predict(encoded, verbose=0)[0]
            safe = _safe_moves(state, probs)

            for i, (prob, move) in enumerate(safe[:beam_width * 2]):
                next_state = state.apply_move(move)
                key = next_state.facelets.tobytes()
                if key in visited:
                    continue  # Skip loops
                new_visited = visited | {key}
                new_score = score + math.log(max(prob, 1e-10))
                candidates.append((new_score, next_state, moves + [move], new_visited))

        if not candidates:
            break

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: -x[0])
        beams = candidates[:beam_width]

        # Check if any beam is solved
        for score, state, moves, visited in beams:
            if is_f2l_solved(state):
                return moves, True, "solved"

    # Return best beam result
    best = beams[0]
    solved = is_f2l_solved(best[1])
    return best[2], solved, "solved" if solved else "move_cap"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def nn_solve(
    model,
    cube: CubeState,
    max_moves: int = MAX_MOVES,
    use_beam: bool = True,
    beam_width: int = BEAM_WIDTH,
) -> tuple[list[str], bool, str]:
    """
    Run the neural network solver with all safety mechanisms.

    Returns:
        moves       : list of moves applied
        solved      : whether F2L is solved
        exit_reason : 'solved' | 'loop_detected' | 'move_cap' | 'no_safe_moves'
    """
    original = cube.copy()

    if use_beam and beam_width > 1:
        moves, solved, reason = beam_solve(model, cube, max_moves, beam_width)
    else:
        moves, solved, reason = greedy_solve(model, cube, max_moves)

    # Verify cross is still intact after NN moves
    result_state = original.apply_moves(moves)
    if not is_cross_solved(result_state):
        # Cross was broken despite our guard — return original state, no moves
        return [], False, "cross_broken"

    return moves, solved, reason
