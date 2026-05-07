"""
cube/f2l_checker.py
-------------------
Checks whether the First Two Layers (F2L) are solved.

F2L is solved when:
  1. The cross (D-face edges) is solved.
  2. All 4 corner-edge pairs in the first two layers are correctly placed
     and oriented.

Facelet indices reference (see cube/state.py for full layout):

  Cross edges (D face edges — center of each D-edge row):
    D-F edge: 46 (D), 25 (F)
    D-R edge: 50 (D), 34 (R)
    D-B edge: 52 (D), 43 (B)
    D-L edge: 48 (D), 16 (L)

  D-face corners (and their U-layer partner facelets):
    DFR corner: 45(D), 26(F), 33(R)   ... wait, let me use correct indices:
    Corners (D layer):
      DFL: facelets 51(D), 17(L), 24(F)
      DFR: facelets 47(D), 26(F), 33(R)   -- actually index 47 is D[0][2]? 
      Let me restate:

D face grid:
  45 46 47
  48 49 50
  51 52 53

So corners of D face:
  DFL = indices 51(D), 15(L-bot-left... wait need to be precise)

Correct mapping from the face layout:
  L face (9-17):  row0=9,10,11  row1=12,13,14  row2=15,16,17
  F face (18-26): row0=18,19,20  row1=21,22,23  row2=24,25,26
  R face (27-35): row0=27,28,29  row1=30,31,32  row2=33,34,35
  B face (36-44): row0=36,37,38  row1=39,40,41  row2=42,43,44
  D face (45-53): row0=45,46,47  row1=48,49,50  row2=51,52,53

4 F2L corner-edge pairs and their facelets:
  Slot FR: corner DFR (47,26,33) + edge FR (23,32) -- middle row F-right, R-left
  Slot FL: corner DFL (45,24,17) -- wait, let me redo corners carefully.

D-face corners (viewed from above, WCA standard):
  UFR corner at bottom = DFR: D[0][2]=47, F[2][2]=26, R[2][0]=33
  UFL corner at bottom = DFL: D[0][0]=45, F[2][0]=24, L[2][2]=17
  UBL corner at bottom = DBL: D[2][0]=51, B[2][2]=44, L[2][0]=15
  UBR corner at bottom = DBR: D[2][2]=53, B[2][0]=42, R[2][2]=35

Middle-layer edges (F2L edges):
  FR edge: F[1][2]=23, R[1][0]=30
  FL edge: F[1][0]=21, L[1][2]=14
  BL edge: B[1][2]=41, L[1][0]=12
  BR edge: B[1][0]=39, R[1][2]=32

D-face center: 49  (color 5=yellow)
Cross edges:
  DF edge: D[0][1]=46, F[2][1]=25
  DR edge: D[1][2]=50, R[2][1]=34
  DB edge: D[2][1]=52, B[2][1]=43
  DL edge: D[1][0]=48, L[2][1]=16
"""

import numpy as np
from .state import CubeState, SOLVED_STATE

# ── Facelet groups ────────────────────────────────────────────────────────────

# Cross: D-center + 4 cross edges (must all be correct color)
CROSS_FACELETS = [
    49,       # D center
    46, 25,   # DF edge
    50, 34,   # DR edge
    52, 43,   # DB edge
    48, 16,   # DL edge
]

# Face centers (always fixed on a standard cube)
CENTERS = [4, 13, 22, 31, 40, 49]  # U L F R B D

# F2L slots: each is (corner_facelets, edge_facelets)
# corner = 3 facelets, edge = 2 facelets
F2L_SLOTS = {
    "FR": {
        "corner": [47, 26, 33],   # DFR
        "edge":   [23, 30],        # FR middle edge
    },
    "FL": {
        "corner": [45, 24, 17],   # DFL
        "edge":   [21, 14],        # FL middle edge
    },
    "BL": {
        "corner": [51, 44, 15],   # DBL
        "edge":   [41, 12],        # BL middle edge
    },
    "BR": {
        "corner": [53, 42, 35],   # DBR
        "edge":   [39, 32],        # BR middle edge
    },
}

# What colors each slot should have in the solved state
def _solved_colors(indices):
    return [int(SOLVED_STATE[i]) for i in indices]

SLOT_SOLVED_COLORS = {
    slot: {
        "corner": _solved_colors(data["corner"]),
        "edge":   _solved_colors(data["edge"]),
    }
    for slot, data in F2L_SLOTS.items()
}


# ── Public API ────────────────────────────────────────────────────────────────

def is_cross_solved(cube: CubeState) -> bool:
    """True if the D-layer cross is solved (correct colors at all 8 cross facelets)."""
    for idx in CROSS_FACELETS:
        if cube.facelets[idx] != SOLVED_STATE[idx]:
            return False
    return True


def is_slot_solved(cube: CubeState, slot: str) -> bool:
    """True if a specific F2L slot (FR/FL/BL/BR) is solved."""
    data = F2L_SLOTS[slot]
    solved = SLOT_SOLVED_COLORS[slot]
    for i, idx in enumerate(data["corner"]):
        if cube.facelets[idx] != solved["corner"][i]:
            return False
    for i, idx in enumerate(data["edge"]):
        if cube.facelets[idx] != solved["edge"][i]:
            return False
    return True


def count_solved_slots(cube: CubeState) -> int:
    """Returns how many F2L slots (0-4) are currently solved."""
    return sum(is_slot_solved(cube, slot) for slot in F2L_SLOTS)


def is_f2l_solved(cube: CubeState) -> bool:
    """True if ALL 4 F2L slots are solved (cross + all corner-edge pairs)."""
    if not is_cross_solved(cube):
        return False
    return all(is_slot_solved(cube, slot) for slot in F2L_SLOTS)


def f2l_progress(cube: CubeState) -> dict:
    """Returns a detailed progress report of F2L state."""
    return {
        "cross_solved": is_cross_solved(cube),
        "slots": {slot: is_slot_solved(cube, slot) for slot in F2L_SLOTS},
        "slots_solved": count_solved_slots(cube),
        "f2l_complete": is_f2l_solved(cube),
    }
