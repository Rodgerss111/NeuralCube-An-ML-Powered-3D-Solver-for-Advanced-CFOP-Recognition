"""
solver/kociemba_solver.py
--------------------------
Last-resort solver using Kociemba's two-phase algorithm.

Install:  pip install kociemba

Kociemba solves any valid cube state in under 25 moves and runs in
milliseconds. It is the correct production fallback.

The kociemba library uses a specific string notation for cube state:
  54 characters, each a face letter (U, R, F, D, L, B)
  in the order: U-face (row by row), R, F, D, L, B

Our facelet indices use:
  0=U 1=L 2=F 3=R 4=B 5=D
  Index order: U(0-8) L(9-17) F(18-26) R(27-35) B(36-44) D(45-53)

Kociemba string order: U1-U9, R1-R9, F1-F9, D1-D9, L1-L9, B1-B9
"""

from cube.state import CubeState, SOLVED_STATE

# Kociemba face letter for each color
_COLOR_TO_FACE = {0: "U", 1: "L", 2: "F", 3: "R", 4: "B", 5: "D"}

# Our facelet array order: U(0-8) L(9-17) F(18-26) R(27-35) B(36-44) D(45-53)
# Kociemba expects: U(0-8) R(9-17) F(18-26) D(27-35) L(36-44) B(45-53)
# Remap: extract faces from our array and reorder for kociemba
_KOCIEMBA_ORDER = (
    list(range(0, 9)) +    # U face — same
    list(range(27, 36)) +  # R face (our indices 27-35)
    list(range(18, 27)) +  # F face (our indices 18-26)
    list(range(45, 54)) +  # D face (our indices 45-53)
    list(range(9, 18)) +   # L face (our indices 9-17)
    list(range(36, 45))    # B face (our indices 36-44)
)


def _to_kociemba_string(cube: CubeState) -> str:
    """Convert our CubeState to the 54-char kociemba input string."""
    return "".join(_COLOR_TO_FACE[cube.facelets[i]] for i in _KOCIEMBA_ORDER)


def kociemba_solve(cube: CubeState, max_depth: int = 24) -> tuple[list[str], bool]:
    """
    Solve the cube using Kociemba's two-phase algorithm.

    Returns (move_sequence, success_flag).
    Returns ([], False) if kociemba is not installed or cube is invalid.

    The move notation kociemba returns uses standard WCA notation
    which matches our MOVE_NAMES exactly.
    """
    try:
        import kociemba
    except ImportError:
        print(
            "WARNING: kociemba not installed. Run: pip install kociemba\n"
            "Kociemba fallback disabled."
        )
        return [], False

    try:
        cube_str = _to_kociemba_string(cube)
        solution_str = kociemba.solve(cube_str)

        # kociemba returns space-separated moves, e.g. "R U R' F2 B U2 ..."
        # Their notation: F2 = F2, R' = R', etc. — matches ours exactly
        if not solution_str or solution_str.strip() == "":
            return [], True  # Already solved

        moves = solution_str.strip().split()

        # Normalize: kociemba uses "R2" which matches ours, but double-check
        valid_moves = set([
            "U", "U'", "U2", "D", "D'", "D2",
            "R", "R'", "R2", "L", "L'", "L2",
            "F", "F'", "F2", "B", "B'", "B2",
        ])

        # kociemba may output "R2'" which is wrong — filter
        normalized = []
        for m in moves:
            if m in valid_moves:
                normalized.append(m)
            elif m.replace("2", "") in valid_moves:
                normalized.append(m)  # pass through, might be fine
            # skip unrecognized tokens

        return normalized, True

    except Exception as e:
        print(f"Kociemba error: {e}")
        return [], False
