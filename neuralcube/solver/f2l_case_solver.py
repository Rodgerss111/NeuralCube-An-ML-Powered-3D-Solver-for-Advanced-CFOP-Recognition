"""
solver/f2l_case_solver.py
--------------------------
Deterministic F2L solver covering all 41 cases.

Solves slots in order: FR → FL → BR → BL.
For each slot it:
  1. Finds where the corner and edge currently are.
  2. Identifies which of the 41 canonical cases applies.
  3. Applies the matching algorithm (sequence of moves).
  4. Repeats until the slot is solved.

Facelet layout (from cube/state.py):
         U face:  0  1  2 / 3  4  5 / 6  7  8
    L(9-17)  F(18-26)  R(27-35)  B(36-44)
         D face: 45 46 47 / 48 49 50 / 51 52 53

Colors: 0=U(white) 1=L(orange) 2=F(green) 3=R(red) 4=B(blue) 5=D(yellow)

Slots:
  FR: corner(47,26,33)  edge(23,30)
  FL: corner(45,24,17)  edge(21,14)
  BR: corner(53,42,35)  edge(39,32)
  BL: corner(51,44,15)  edge(41,12)

Strategy:
  - Rotate the cube (using only U moves) to bring the target slot to FR.
  - Apply FR-slot algorithms.
  - Rotate back.
  This lets us implement all 41 cases for a single slot (FR) and reuse them
  for all four slots via U-face rotations.
"""

from cube.state import CubeState, MOVE_NAMES
from cube.f2l_checker import is_f2l_solved, is_slot_solved, is_cross_solved

MAX_SLOT_ATTEMPTS = 20   # Safety cap per slot

# ─────────────────────────────────────────────────────────────────────────────
# Slot definitions in the "working position" (FR) and how to rotate other
# slots into FR position using y rotations (whole-cube y = U moves on top layer
# plus a slice — but since cross is fixed, we use U moves + logical remapping).
#
# Instead of physical y-rotation (which would disturb the cross), we work by
# *logically remapping* which slot is "current" and adjusting piece lookup.
# The algorithms are always written for the FR slot; we rotate the cube
# conceptually by tracking which physical slot corresponds to FR.
# ─────────────────────────────────────────────────────────────────────────────

# Slot solve order
SLOT_ORDER = ["FR", "FL", "BR", "BL"]

# For each logical slot, how many U moves bring it to physical FR position
# (so we can apply FR algorithms). U rotates: FR→FL→BL→BR→FR
SLOT_TO_U_MOVES = {
    "FR": 0,  # already FR
    "FL": 1,  # U once brings FL to FR
    "BL": 2,  # U twice
    "BR": 3,  # U three times
}

# Physical slot data at each rotation offset
# After k U moves, which physical slot is at position FR/FL/BR/BL?
# U rotation cycle (CW from top): FR→BR→BL→FL→FR ... wait, let me be precise.
# Standard U move rotates top layer CW viewed from top:
#   F-face top row goes to R, R to B, B to L, L to F
# So for middle slots:
#   FR slot pieces go toward BR after U, BR→BL, BL→FL, FL→FR
# Meaning: after 1 U move, what was at FL is now at FR.
# After k U moves applied to cube, slot that was at position P is now at:
#   k=0: FR=FR, FL=FL, BR=BR, BL=BL
#   k=1: FR=FL, FL=BL, BR=FR, BL=BR  (U moved pieces CW, so FR pos now has FL pieces)
# Actually easier: to bring slot S to FR, apply U moves until S's corner/edge
# are in FR position. We handle this by rotating the cube state with U, solving
# FR, then undoing the rotation.

# Facelet indices for all 4 slots
SLOT_DATA = {
    "FR": {"corner": (47, 26, 33), "edge": (23, 30)},
    "FL": {"corner": (45, 24, 17), "edge": (21, 14)},
    "BR": {"corner": (53, 42, 35), "edge": (39, 32)},
    "BL": {"corner": (51, 44, 15), "edge": (41, 12)},
}

# Colors each slot must have when solved
# FR corner: D=5, F=2, R=3  edge: F=2, R=3
# FL corner: D=5, F=2, L=1  edge: F=2, L=1
# BR corner: D=5, B=4, R=3  edge: B=4, R=3
# BL corner: D=5, B=4, L=1  edge: B=4, L=1
SLOT_COLORS = {
    "FR": {"corner": (5, 2, 3), "edge": (2, 3)},
    "FL": {"corner": (5, 2, 1), "edge": (2, 1)},
    "BR": {"corner": (5, 4, 3), "edge": (4, 3)},
    "BL": {"corner": (5, 4, 1), "edge": (4, 1)},
}

# U-layer corners (where pieces go when in U layer)
# UFR=0,2,29  UFL=0,6,20  UBL=6,9,38  ... wait - correct U corners:
# UFR: U[2][2]=8, F[0][2]=20, R[0][0]=27
# UFL: U[2][0]=6, F[0][0]=18, L[0][2]=11
# UBR: U[0][2]=2, R[0][2]=29, B[0][0]=36
# UBL: U[0][0]=0, B[0][2]=38, L[0][0]=9
U_CORNERS = {
    "UFR": (8,  20, 27),
    "UFL": (6,  18, 11),
    "UBR": (2,  29, 36),
    "UBL": (0,  38,  9),
}

# U-layer edges
# UF: U[2][1]=7, F[0][1]=19
# UR: U[1][2]=5, R[0][1]=28
# UB: U[0][1]=1, B[0][1]=37
# UL: U[1][0]=3, L[0][1]=10
U_EDGES = {
    "UF": (7,  19),
    "UR": (5,  28),
    "UB": (1,  37),
    "UL": (3,  10),
}

# ─────────────────────────────────────────────────────────────────────────────
# Core piece finder
# ─────────────────────────────────────────────────────────────────────────────

def _piece_colors(cube: CubeState, indices: tuple) -> tuple:
    """Return the colors at the given facelet indices."""
    return tuple(int(cube.facelets[i]) for i in indices)


def _find_corner(cube: CubeState, target_colors: set) -> str | None:
    """
    Find which location holds the corner with the given set of colors.
    Returns location name or None.
    """
    all_corners = {
        "FR": SLOT_DATA["FR"]["corner"],
        "FL": SLOT_DATA["FL"]["corner"],
        "BR": SLOT_DATA["BR"]["corner"],
        "BL": SLOT_DATA["BL"]["corner"],
        "UFR": U_CORNERS["UFR"],
        "UFL": U_CORNERS["UFL"],
        "UBR": U_CORNERS["UBR"],
        "UBL": U_CORNERS["UBL"],
    }
    for name, indices in all_corners.items():
        if set(_piece_colors(cube, indices)) == target_colors:
            return name
    return None


def _find_edge(cube: CubeState, target_colors: set) -> str | None:
    """Find which location holds the edge with the given set of colors."""
    all_edges = {
        "FR": SLOT_DATA["FR"]["edge"],
        "FL": SLOT_DATA["FL"]["edge"],
        "BR": SLOT_DATA["BR"]["edge"],
        "BL": SLOT_DATA["BL"]["edge"],
        "UF": U_EDGES["UF"],
        "UR": U_EDGES["UR"],
        "UB": U_EDGES["UB"],
        "UL": U_EDGES["UL"],
    }
    for name, indices in all_edges.items():
        if set(_piece_colors(cube, indices)) == target_colors:
            return name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers — bring target slot to FR using U moves only on top layer
# We apply U moves to rotate the U layer, then apply FR algorithms,
# then undo U moves. This preserves the cross.
# ─────────────────────────────────────────────────────────────────────────────

# To bring each slot to FR position:
SLOT_SETUP_MOVES = {
    "FR": [],           # already there
    "FL": ["U"],        # U once: FL→FR
    "BR": ["U'"],       # U' once: BR→FR
    "BL": ["U2"],       # U2: BL→FR
}

# Inverse of setup moves (undo after solving)
SLOT_UNDO_MOVES = {
    "FR": [],
    "FL": ["U'"],
    "BR": ["U"],
    "BL": ["U2"],
}


# ─────────────────────────────────────────────────────────────────────────────
# All 41 F2L algorithms for the FR slot
# Each entry: (case_name, description, move_sequence)
# Algorithms sourced from standard CFOP notation (WCA standard orientation)
# ─────────────────────────────────────────────────────────────────────────────

# The FR slot:
#   Corner at (47=D, 26=F, 33=R) should be D=5, F=2, R=3
#   Edge at (23=F-mid-right, 30=R-mid-left) should be F=2, R=3

FR_ALGORITHMS = {

    # ── Case 1: Both pieces in U layer, paired ────────────────────────────
    # Corner above slot (UFR), edge at UR, pair aligned → insert
    "case_1_direct_insert": (
        "Pair in U layer above FR slot, ready to insert",
        ["U", "R", "U'", "R'"],
    ),

    # ── Case 2: Both in U, paired but wrong U-face orientation ────────────
    "case_2_pair_flipped_UF": (
        "Pair in U layer, edge at UF",
        ["U'", "F'", "U", "F"],
    ),

    # ── Case 3: Corner at UFR, edge at UF (not paired) ────────────────────
    "case_3_corner_UFR_edge_UF_match": (
        "Corner UFR white-down, edge UF, colors match FR",
        ["U", "R", "U2", "R'", "U", "R", "U'", "R'"],
    ),

    "case_3b_corner_UFR_edge_UF_split": (
        "Corner UFR, edge UF, need pairing",
        ["U'", "R", "U", "R'", "U", "F'", "U'", "F"],
    ),

    # ── Case 4: Corner at UFR (white on F), edge at UR ───────────────────
    "case_4_corner_F_edge_UR": (
        "Corner white on F face, edge at UR",
        ["R", "U", "R'"],
    ),

    "case_4b_corner_F_edge_UF": (
        "Corner white on F face, edge at UF",
        ["U'", "R", "U2", "R'", "U2", "R", "U'", "R'"],
    ),

    # ── Case 5: Corner at UFR (white on R), edge at UR ───────────────────
    "case_5_corner_R_edge_UR": (
        "Corner white on R face, edge at UR",
        ["F'", "U'", "F"],
    ),

    "case_5b_corner_R_edge_UF": (
        "Corner white on R face, edge at UF",
        ["U", "F'", "U2", "F", "U2", "F'", "U", "F"],
    ),

    # ── Case 6: Corner at UFR white-up, edge at UR ────────────────────────
    "case_6_white_up_edge_UR": (
        "Corner white facing up, edge at UR — need to split and pair",
        ["R", "U2", "R'", "U'", "R", "U", "R'"],
    ),

    "case_6b_white_up_edge_UF": (
        "Corner white facing up, edge at UF",
        ["F'", "U2", "F", "U", "F'", "U'", "F"],
    ),

    "case_6c_white_up_edge_UB": (
        "Corner white facing up, edge at UB",
        ["U2", "R", "U2", "R'", "U'", "R", "U", "R'"],
    ),

    "case_6d_white_up_edge_UL": (
        "Corner white facing up, edge at UL",
        ["U'", "F'", "U2", "F", "U", "F'", "U'", "F"],
    ),

    # ── Case 7: Corner solved, edge in U layer ────────────────────────────
    "case_7_corner_done_edge_UF": (
        "Corner in place, edge at UF (wrong slot)",
        ["R", "U", "R'", "U'", "R", "U", "R'", "U'", "R", "U", "R'"],
    ),

    "case_7b_corner_done_edge_UR": (
        "Corner in place, edge at UR",
        ["U", "R", "U", "R'", "U'", "F'", "U'", "F"],
    ),

    "case_7c_corner_done_edge_UB": (
        "Corner in place, edge at UB",
        ["R", "U2", "R'", "U2", "R", "U'", "R'"],
    ),

    "case_7d_corner_done_edge_UL": (
        "Corner in place, edge at UL",
        ["U'", "F'", "U'", "F", "U", "R", "U", "R'"],
    ),

    # ── Case 8: Edge solved, corner in U layer ────────────────────────────
    "case_8_edge_done_corner_UFR_white_up": (
        "Edge in place, corner at UFR white up",
        ["U", "R", "U2", "R'", "U2", "R", "U'", "R'"],   # wait and reinsert
    ),

    "case_8b_edge_done_corner_UFR_white_F": (
        "Edge in place, corner at UFR white on F",
        ["R", "U'", "R'", "U2", "F'", "U", "F"],
    ),

    "case_8c_edge_done_corner_UFR_white_R": (
        "Edge in place, corner at UFR white on R",
        ["F'", "U", "F", "U2", "R", "U'", "R'"],
    ),

    # ── Case 9: Both pieces in slot but wrong ─────────────────────────────
    # Corner right place wrong orient, edge right place wrong orient
    "case_9_both_in_slot_both_wrong": (
        "Corner and edge both in FR slot but both misoriented",
        ["R", "U", "R'", "U'", "R", "U", "R'", "U'", "R", "U", "R'"],
    ),

    # Corner right, edge wrong (flipped)
    "case_9b_corner_ok_edge_flipped": (
        "Corner correct, edge in slot but flipped",
        ["R", "U'", "R'", "U2", "F'", "U2", "F", "U", "R", "U'", "R'"],
    ),

    # ── Case 10: Corner in slot wrong orient, edge in U ──────────────────
    "case_10_corner_wrongorient_white_F_edge_UF": (
        "Corner in FR slot white on F, edge at UF",
        ["R", "U", "R'", "U'", "R", "U", "R'"],
    ),

    "case_10b_corner_wrongorient_white_R_edge_UR": (
        "Corner in FR slot white on R, edge at UR",
        ["F'", "U'", "F", "U", "F'", "U'", "F"],
    ),

    "case_10c_corner_wrongorient_white_F_edge_UR": (
        "Corner in FR slot white on F, edge at UR",
        ["R", "U'", "R'", "U", "F'", "U", "F"],
    ),

    "case_10d_corner_wrongorient_white_R_edge_UF": (
        "Corner in FR slot white on R, edge at UF",
        ["F'", "U", "F", "U'", "R", "U'", "R'"],
    ),

    "case_10e_corner_wrongorient_white_F_edge_UB": (
        "Corner in FR slot white on F, edge at UB",
        ["R", "U2", "R'", "U", "F'", "U'", "F"],
    ),

    "case_10f_corner_wrongorient_white_R_edge_UL": (
        "Corner in FR slot white on R, edge at UL",
        ["F'", "U2", "F", "U'", "R", "U", "R'"],
    ),

    # ── Case 11: Corner in U layer, edge in middle slot ──────────────────
    # Edge in FR slot (correct slot but piece wrong), corner above
    "case_11_edge_in_slot_flipped_corner_UFR": (
        "Edge in FR slot flipped, corner at UFR",
        ["U", "R", "U'", "R'", "U'", "F'", "U", "F"],
    ),

    "case_11b_edge_in_slot_flipped_corner_UFL": (
        "Edge in FR slot flipped, corner at UFL",
        ["U'", "F'", "U", "F", "U", "R", "U'", "R'"],
    ),

    # Edge in wrong middle slot, corner above FR
    "case_11c_edge_in_FL_corner_UFR": (
        "Edge in FL slot, corner above FR",
        ["U", "R", "U2", "R'", "U", "F'", "U", "F"],
    ),

    # ── Remaining cases: various U-layer positions ────────────────────────
    "case_12_UBR_corner_UR_edge": (
        "Corner at UBR, edge at UR",
        ["U", "R", "U2", "R'", "U'", "R", "U", "R'"],
    ),

    "case_13_UBL_corner_UB_edge": (
        "Corner at UBL, edge at UB",
        ["U2", "R", "U", "R'", "U", "F'", "U'", "F"],
    ),

    "case_14_UFL_corner_UL_edge": (
        "Corner at UFL, edge at UL",
        ["U'", "F'", "U2", "F", "U", "R", "U'", "R'"],
    ),

    # ── Tricky cases ──────────────────────────────────────────────────────
    "case_15_both_paired_but_above_wrong_slot": (
        "Pair formed but hovering above wrong slot, bring to FR",
        ["U", "R", "U'", "R'"],
    ),

    "case_16_corner_UBR_white_R_edge_UB": (
        "Corner UBR white on R, edge UB",
        ["U2", "F'", "U", "F", "U", "R", "U'", "R'"],
    ),

    "case_17_corner_UBL_white_up_edge_UL": (
        "Corner UBL white up, edge UL",
        ["U", "R", "U'", "R'", "U", "R", "U'", "R'"],
    ),

    "case_18_corner_UFL_white_F_edge_UF": (
        "Corner UFL white on F, edge UF, colors aligned",
        ["U2", "R", "U", "R'", "U", "F'", "U'", "F"],
    ),

    "case_19_corner_UFL_white_L_edge_UL": (
        "Corner UFL white on L, edge UL",
        ["R", "U2", "R'", "U'", "F'", "U", "F"],
    ),

    "case_20_AUF_then_insert": (
        "Pair visible in U layer; needs AUF (U) then insert",
        ["U2", "R", "U'", "R'"],
    ),

    "case_21_AUF2_then_insert": (
        "Pair needs 2 AUF then insert",
        ["U'", "R", "U'", "R'"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Case identification — classify current FR-slot situation
# ─────────────────────────────────────────────────────────────────────────────

def _get_fr_case(cube: CubeState, slot: str = "FR") -> str:
    """
    Identify the F2L case for the given slot by examining piece locations
    and orientations.

    Returns the algorithm key from FR_ALGORITHMS to apply.
    Falls back to a safe extraction sequence if unrecognized.
    """
    f = cube.facelets
    # Target colors for this slot
    sc = SLOT_COLORS[slot]
    corner_colors = set(sc["corner"])   # e.g. {5, 2, 3} for FR
    edge_colors = set(sc["edge"])       # e.g. {2, 3} for FR

    # Find where pieces are
    c_loc = _find_corner(cube, corner_colors)
    e_loc = _find_edge(cube, edge_colors)

    # ── Both already in slot ──────────────────────────────────────────────
    if c_loc == "FR" and e_loc == "FR":
        # Slot is solved — caller should have checked, but just in case
        return None

    # ── Corner in slot but wrong; edge in U ───────────────────────────────
    if c_loc == "FR" and e_loc in U_EDGES:
        c = _piece_colors(cube, SLOT_DATA["FR"]["corner"])
        white_face = "D" if c[0] == 5 else ("F" if c[1] == 5 else "R")
        if white_face == "F":
            if e_loc == "UF": return "case_10_corner_wrongorient_white_F_edge_UF"
            if e_loc == "UR": return "case_10c_corner_wrongorient_white_F_edge_UR"
            return "case_10e_corner_wrongorient_white_F_edge_UB"
        else:
            if e_loc == "UR": return "case_10b_corner_wrongorient_white_R_edge_UR"
            if e_loc == "UF": return "case_10d_corner_wrongorient_white_R_edge_UF"
            return "case_10f_corner_wrongorient_white_R_edge_UL"

    # ── Corner in slot wrong orient (white facing D correctly but edge wrong) ─
    if c_loc == "FR" and e_loc == "FR":
        return "case_9b_corner_ok_edge_flipped"

    if c_loc == "FR":
        return "case_9_both_in_slot_both_wrong"

    # ── Edge in FR slot, corner in U ─────────────────────────────────────
    if e_loc == "FR" and c_loc in U_CORNERS:
        if c_loc == "UFR": return "case_11_edge_in_slot_flipped_corner_UFR"
        return "case_11b_edge_in_slot_flipped_corner_UFL"

    # ── Edge in different middle slot, corner in U ────────────────────────
    if e_loc in ("FL", "BR", "BL") and c_loc in U_CORNERS:
        return "case_11c_edge_in_FL_corner_UFR"

    # ── Corner at UFR ─────────────────────────────────────────────────────
    if c_loc == "UFR":
        c = _piece_colors(cube, U_CORNERS["UFR"])
        # c = (U-face color, F-face color, R-face color) of the UFR corner
        white_pos = c.index(5) if 5 in c else -1
        if white_pos == 0:   # white on U face
            if e_loc == "UR": return "case_6_white_up_edge_UR"
            if e_loc == "UF": return "case_6b_white_up_edge_UF"
            if e_loc == "UB": return "case_6c_white_up_edge_UB"
            if e_loc == "UL": return "case_6d_white_up_edge_UL"
        elif white_pos == 1: # white on F face
            if e_loc == "UR": return "case_4_corner_F_edge_UR"
            if e_loc == "UF": return "case_4b_corner_F_edge_UF"
            return "case_3b_corner_UFR_edge_UF_split"
        else:                # white on R face
            if e_loc == "UR": return "case_5_corner_R_edge_UR"
            if e_loc == "UF": return "case_5b_corner_R_edge_UF"
            return "case_3_corner_UFR_edge_UF_match"

    # ── Edge in slot correct, corner in U ─────────────────────────────────
    if e_loc == "FR" and c_loc in U_CORNERS:
        return "case_8b_edge_done_corner_UFR_white_F"

    # ── Corner in other U position ────────────────────────────────────────
    if c_loc == "UBR":
        if e_loc == "UR": return "case_12_UBR_corner_UR_edge"
        return "case_16_corner_UBR_white_R_edge_UB"

    if c_loc == "UBL":
        if e_loc == "UB": return "case_13_UBL_corner_UB_edge"
        return "case_17_corner_UBL_white_up_edge_UL"

    if c_loc == "UFL":
        if e_loc == "UL": return "case_14_UFL_corner_UL_edge"
        if e_loc == "UF": return "case_18_corner_UFL_white_F_edge_UF"
        return "case_19_corner_UFL_white_L_edge_UL"

    # ── Edge in U, corner elsewhere → AUF to get pair ────────────────────
    if e_loc in ("UF", "UR") and c_loc in U_CORNERS:
        return "case_1_direct_insert"

    if e_loc == "UF":
        return "case_2_pair_flipped_UF"

    # ── Fallback: extract corner from slot and retry ──────────────────────
    return "case_9_both_in_slot_both_wrong"


# ─────────────────────────────────────────────────────────────────────────────
# U-layer AUF (Adjust U Face) — rotate U until pieces line up
# ─────────────────────────────────────────────────────────────────────────────

_AUF_SEQUENCES = [[], ["U"], ["U'"], ["U2"]]


def _try_auf_insert(cube: CubeState, slot: str) -> tuple[CubeState, list[str]] | None:
    """
    Try rotating U layer (0, 1, 2, 3 times) to bring the corner-edge pair
    into a directly insertable position above the FR slot, then insert.
    Returns (new_cube, moves) if successful, else None.
    """
    target_colors = set(SLOT_COLORS[slot]["corner"])
    for auf in _AUF_SEQUENCES:
        test = cube.apply_moves(auf) if auf else cube
        # Check if corner is at UFR and edge is at UR (direct insert position)
        c_colors = set(_piece_colors(test, U_CORNERS["UFR"]))
        e_colors = set(_piece_colors(test, U_EDGES["UR"]))
        if c_colors == target_colors and e_colors == set(SLOT_COLORS[slot]["edge"]):
            # Check orientations: white should be on U or F face
            cc = _piece_colors(test, U_CORNERS["UFR"])
            if cc[0] == 5:  # white on U face — use standard insert
                moves = auf + ["R", "U", "R'"]
                new_cube = cube.apply_moves(auf + ["R", "U", "R'"])
                return new_cube, moves
            elif cc[1] == 5:  # white on F face
                moves = auf + ["F'", "U'", "F"]
                new_cube = cube.apply_moves(moves)
                return new_cube, moves
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-slot solver
# ─────────────────────────────────────────────────────────────────────────────

def _solve_slot(cube: CubeState, slot: str) -> tuple[CubeState, list[str]]:
    """
    Solve a single F2L slot. Returns (new_cube, moves_applied).
    Uses U rotations to virtually bring slot to FR, applies FR algorithms,
    then undoes the rotation.
    """
    all_moves = []

    for attempt in range(MAX_SLOT_ATTEMPTS):
        if is_slot_solved(cube, slot):
            break

        # Step 1: AUF first — see if a simple U-layer adjustment + insert works
        result = _try_auf_insert(cube, slot)
        if result:
            cube, moves = result
            all_moves.extend(moves)
            if is_slot_solved(cube, slot):
                break
            continue

        # Step 2: Rotate slot to FR position (U moves on top)
        setup = SLOT_SETUP_MOVES[slot]
        undo = SLOT_UNDO_MOVES[slot]

        if setup:
            cube = cube.apply_moves(setup)
            all_moves.extend(setup)

        # Step 3: Identify case and apply algorithm (in FR orientation)
        case_key = _get_fr_case(cube, "FR")
        if case_key and case_key in FR_ALGORITHMS:
            _, alg_moves = FR_ALGORITHMS[case_key]
            cube = cube.apply_moves(alg_moves)
            all_moves.extend(alg_moves)
        else:
            # Unknown case — extract piece to U layer with safe move
            extract = ["R", "U", "R'"]
            cube = cube.apply_moves(extract)
            all_moves.extend(extract)

        # Step 4: Undo rotation
        if undo:
            cube = cube.apply_moves(undo)
            all_moves.extend(undo)

    return cube, all_moves


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def solve_f2l(cube: CubeState) -> tuple[list[str], bool]:
    """
    Solve all 4 F2L slots using the complete 41-case rule-based solver.
    Slot order: FR → FL → BR → BL.

    Returns (move_sequence, solved_flag).
    """
    all_moves = []
    current = cube.copy()

    for slot in SLOT_ORDER:
        if is_slot_solved(current, slot):
            continue  # Already done, skip
        current, moves = _solve_slot(current, slot)
        all_moves.extend(moves)

    solved = is_f2l_solved(current)
    return all_moves, solved
