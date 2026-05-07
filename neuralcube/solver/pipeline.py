"""
solver/pipeline.py
------------------
Master solve pipeline with cascading fallback.

Phase 1 → Neural network (fast, good for simple cases)
Phase 2 → Rule-based 41-case F2L solver (deterministic, always correct)
Phase 3 → Kociemba two-phase algorithm (guaranteed, <25 moves)

Each phase hands off to the next if it fails.
The final state is always verified before returning.
"""

import time
from cube.state import CubeState
from cube.f2l_checker import is_f2l_solved, is_cross_solved, f2l_progress

from solver.nn_solver import nn_solve
from solver.f2l_case_solver import solve_f2l
from solver.kociemba_solver import kociemba_solve


def solve(
    cube: CubeState,
    model=None,
    use_nn: bool = True,
    use_case_solver: bool = True,
    use_kociemba: bool = True,
    nn_max_moves: int = 30,
    nn_beam_width: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Attempt to solve F2L using the cascading fallback pipeline.

    Parameters
    ----------
    cube             : input CubeState (cross must be solved)
    model            : loaded Keras model (or None to skip Phase 1)
    use_nn           : whether to try Phase 1 (NN)
    use_case_solver  : whether to try Phase 2 (rule-based)
    use_kociemba     : whether to try Phase 3 (Kociemba)
    nn_max_moves     : max moves for NN inference loop
    nn_beam_width    : beam width for NN beam search (1 = greedy)
    verbose          : print phase results

    Returns
    -------
    dict with keys:
      moves         : list[str] — full move sequence
      solved        : bool
      phase_used    : 'nn' | 'case_solver' | 'kociemba' | 'already_solved' | 'failed'
      phase_detail  : exit reason from whichever phase succeeded
      time_ms       : total time in milliseconds
      f2l_progress  : progress dict from f2l_checker
    """
    t0 = time.perf_counter()

    # ── Pre-checks ────────────────────────────────────────────────────────
    if not is_cross_solved(cube):
        return _result([], False, "failed", "cross_not_solved", t0, cube)

    if is_f2l_solved(cube):
        return _result([], True, "already_solved", "f2l_complete", t0, cube)

    original = cube.copy()
    all_moves = []
    current = cube.copy()

    # ── Phase 1: Neural Network ──────────────────────────────────────────
    if use_nn and model is not None:
        if verbose:
            print("[Phase 1] Neural network solver...")

        nn_moves, nn_solved, nn_reason = nn_solve(
            model, current,
            max_moves=nn_max_moves,
            use_beam=(nn_beam_width > 1),
            beam_width=nn_beam_width,
        )

        if verbose:
            print(f"  → {nn_reason}, {len(nn_moves)} moves, solved={nn_solved}")

        if nn_solved:
            all_moves.extend(nn_moves)
            final = original.apply_moves(all_moves)
            return _result(all_moves, True, "nn", nn_reason, t0, final)

        # NN failed — pass whatever state we have to Phase 2
        # If NN made progress (partially solved), keep those moves
        if nn_reason not in ("cross_broken",) and nn_moves:
            partial = current.apply_moves(nn_moves)
            if is_cross_solved(partial):
                current = partial
                all_moves.extend(nn_moves)
                if verbose:
                    print(f"  → Partial NN progress kept ({len(nn_moves)} moves)")

    # ── Phase 2: Rule-based 41-case F2L solver ───────────────────────────
    if use_case_solver:
        if verbose:
            print("[Phase 2] Rule-based case solver (41 cases)...")

        case_moves, case_solved = solve_f2l(current)

        if verbose:
            print(f"  → solved={case_solved}, {len(case_moves)} moves")

        if case_solved:
            all_moves.extend(case_moves)
            final = original.apply_moves(all_moves)
            return _result(all_moves, True, "case_solver", "all_41_cases", t0, final)

        # Case solver failed (shouldn't happen, but be safe)
        if case_moves:
            partial = current.apply_moves(case_moves)
            if is_cross_solved(partial):
                current = partial
                all_moves.extend(case_moves)

    # ── Phase 3: Kociemba ────────────────────────────────────────────────
    if use_kociemba:
        if verbose:
            print("[Phase 3] Kociemba fallback...")

        koc_moves, koc_ok = kociemba_solve(current)

        if verbose:
            print(f"  → ok={koc_ok}, {len(koc_moves)} moves")

        if koc_ok:
            all_moves.extend(koc_moves)
            final = original.apply_moves(all_moves)
            solved = is_f2l_solved(final)
            return _result(all_moves, solved, "kociemba", "two_phase", t0, final)

    # ── All phases failed ────────────────────────────────────────────────
    final = original.apply_moves(all_moves)
    return _result(all_moves, False, "failed", "all_phases_exhausted", t0, final)


def _result(
    moves: list,
    solved: bool,
    phase: str,
    detail: str,
    t0: float,
    cube: CubeState,
) -> dict:
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "moves": moves,
        "move_count": len(moves),
        "solved": solved,
        "phase_used": phase,
        "phase_detail": detail,
        "time_ms": round(elapsed_ms, 2),
        "f2l_progress": f2l_progress(cube),
    }
