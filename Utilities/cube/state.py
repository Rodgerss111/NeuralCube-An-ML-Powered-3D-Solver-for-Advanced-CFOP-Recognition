"""
cube/state.py
-------------
Cube state as a 54-element integer array (facelets).

Face layout (index into the 54-element array):
         U face
         0  1  2
         3  4  5
         6  7  8
L face       F face       R face       B face
 9 10 11    18 19 20    27 28 29    36 37 38
12 13 14    21 22 23    30 31 32    39 40 41
15 16 17    24 25 26    33 34 35    42 43 44
         D face
         45 46 47
         48 49 50
         51 52 53

Color encoding:
  0 = U (white)
  1 = L (orange)
  2 = F (green)
  3 = R (red)
  4 = B (blue)
  5 = D (yellow)
"""

import numpy as np
import copy

# ── Solved state ────────────────────────────────────────────────────────────
SOLVED_STATE = np.array(
    [0]*9 +   # U
    [1]*9 +   # L
    [2]*9 +   # F
    [3]*9 +   # R
    [4]*9 +   # B
    [5]*9,    # D
    dtype=np.int8
)

# ── Move definitions ─────────────────────────────────────────────────────────
# Each move is a permutation of the 54 facelet indices.
# Generated from first principles; standard WCA orientation.

def _cycle(state, *cycles):
    """Apply a set of 4-cycles to a copy of the state."""
    s = state.copy()
    for cyc in cycles:
        tmp = s[cyc[-1]]
        for i in range(len(cyc)-1, 0, -1):
            s[cyc[i]] = s[cyc[i-1]]
        s[cyc[0]] = tmp
    return s


def _build_move_U(s):
    s = _cycle(s,
        (0,2,8,6), (1,5,7,3),          # U face corners & edges
        (9,36,27,18), (10,37,28,19), (11,38,29,20)  # top row of L,B,R,F
    )
    return s

def _build_move_D(s):
    s = _cycle(s,
        (45,47,53,51), (46,50,52,48),   # D face
        (15,24,33,42), (16,25,34,43), (17,26,35,44)  # bottom row
    )
    return s

def _build_move_R(s):
    s = _cycle(s,
        (27,29,35,33), (28,32,34,30),   # R face
        (2,20,47,38), (5,23,50,41), (8,26,53,44)
    )
    return s

def _build_move_L(s):
    s = _cycle(s,
        (9,11,17,15), (10,14,16,12),    # L face
        (0,36,45,24), (3,39,48,21), (6,42,51,18)
    )
    return s

def _build_move_F(s):
    s = _cycle(s,
        (18,20,26,24), (19,23,25,21),   # F face
        (6,27,47,17), (7,30,46,14), (8,33,45,11)
    )
    return s

def _build_move_B(s):
    s = _cycle(s,
        (36,38,44,42), (37,41,43,39),   # B face
        (0,9,53,29), (1,12,52,32), (2,15,51,35)
    )
    return s


def _apply_twice(fn, s):
    return fn(fn(s))

def _apply_inverse(fn, s):
    # Inverse = apply 3 times (since order-4 permutation)
    return fn(fn(fn(s)))


# ── Move table ───────────────────────────────────────────────────────────────
MOVE_NAMES = [
    "U", "U'", "U2",
    "D", "D'", "D2",
    "R", "R'", "R2",
    "L", "L'", "L2",
    "F", "F'", "F2",
    "B", "B'", "B2",
]

_MOVE_FNS = {
    "U":  _build_move_U,
    "D":  _build_move_D,
    "R":  _build_move_R,
    "L":  _build_move_L,
    "F":  _build_move_F,
    "B":  _build_move_B,
}

def _build_move_table():
    table = {}
    for face, fn in _MOVE_FNS.items():
        table[face]        = fn
        table[face + "'"]  = lambda s, f=fn: _apply_inverse(f, s)
        table[face + "2"]  = lambda s, f=fn: _apply_twice(f, s)
    return table

MOVE_TABLE = _build_move_table()


# ── CubeState class ──────────────────────────────────────────────────────────

class CubeState:
    """
    Represents a Rubik's Cube state as a 54-element numpy int8 array.
    The cross is assumed to be already solved before F2L operations.
    """

    def __init__(self, state: np.ndarray = None):
        if state is None:
            self.facelets = SOLVED_STATE.copy()
        else:
            self.facelets = np.array(state, dtype=np.int8)

    def copy(self) -> "CubeState":
        return CubeState(self.facelets.copy())

    def apply_move(self, move: str) -> "CubeState":
        """Return a new CubeState after applying the given move name."""
        if move not in MOVE_TABLE:
            raise ValueError(f"Unknown move: {move!r}. Valid moves: {MOVE_NAMES}")
        new_facelets = MOVE_TABLE[move](self.facelets)
        return CubeState(new_facelets)

    def apply_moves(self, moves: list) -> "CubeState":
        """Apply a sequence of moves and return the resulting state."""
        state = self
        for m in moves:
            state = state.apply_move(m)
        return state

    def encode(self) -> np.ndarray:
        """
        One-hot encode the 54 facelets into a flat (324,) float32 vector.
        Each facelet becomes a 6-dim one-hot for the 6 colors.
        """
        arr = np.zeros((54, 6), dtype=np.float32)
        arr[np.arange(54), self.facelets] = 1.0
        return arr.flatten()

    def is_solved(self) -> bool:
        return np.array_equal(self.facelets, SOLVED_STATE)

    def __eq__(self, other):
        return np.array_equal(self.facelets, other.facelets)

    def __hash__(self):
        return hash(self.facelets.tobytes())

    def __repr__(self):
        f = self.facelets
        color = {0:"W",1:"O",2:"G",3:"R",4:"B",5:"Y"}
        lines = []
        lines.append("      U")
        for i in range(3):
            lines.append("      " + " ".join(color[f[i*3+j]] for j in range(3)))
        lines.append("L     F     R     B")
        for i in range(3):
            l = " ".join(color[f[9 +i*3+j]] for j in range(3))
            fc= " ".join(color[f[18+i*3+j]] for j in range(3))
            r = " ".join(color[f[27+i*3+j]] for j in range(3))
            b = " ".join(color[f[36+i*3+j]] for j in range(3))
            lines.append(f"{l}   {fc}   {r}   {b}")
        lines.append("      D")
        for i in range(3):
            lines.append("      " + " ".join(color[f[45+i*3+j]] for j in range(3)))
        return "\n".join(lines)
