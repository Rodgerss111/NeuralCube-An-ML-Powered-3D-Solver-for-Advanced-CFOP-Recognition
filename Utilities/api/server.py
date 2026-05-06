"""
api/server.py
-------------
FastAPI REST API for the NeuralCube F2L solver.

Endpoints:
  GET  /          — health check
  GET  /info      — model info
  POST /solve     — solve F2L from a given cube state
  POST /validate  — validate cube state without solving

Run:
  uvicorn api.server:app --reload --port 8000

Request body for /solve:
  {
    "state": [int × 54]   -- facelet array (0=U,1=L,2=F,3=R,4=B,5=D)
  }

Response:
  {
    "moves": ["R", "U", "R'", ...],
    "move_count": 7,
    "solved": true,
    "f2l_progress": { "cross_solved": true, "slots": {...}, ... }
  }
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from typing import Optional, List

from cube.state import CubeState, MOVE_NAMES, SOLVED_STATE
from cube.f2l_checker import is_f2l_solved, is_cross_solved, f2l_progress
from eval.benchmark import greedy_solve

# ── Global model store ────────────────────────────────────────────────────────
_model = None
MODEL_PATH = os.environ.get("MODEL_PATH", "model/saved/f2l_model.h5")
MAX_MOVES = int(os.environ.get("MAX_MOVES", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    global _model
    if os.path.exists(MODEL_PATH):
        from model.network import load_model
        print(f"Loading model from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"WARNING: No model found at {MODEL_PATH}. /solve will return 503.")
    yield


app = FastAPI(
    title="NeuralCube F2L Solver API",
    description="Neural network F2L solver for Rubik's Cube",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow React frontend on any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    state: List[int]

    @field_validator("state")
    @classmethod
    def validate_state_length(cls, v):
        if len(v) != 54:
            raise ValueError(f"State must have exactly 54 facelets, got {len(v)}")
        return v


class SolveResponse(BaseModel):
    moves: List[str]
    move_count: int
    solved: bool
    f2l_progress: dict
    error: Optional[str] = None


class ValidateResponse(BaseModel):
    valid: bool
    cross_solved: bool
    f2l_already_solved: bool
    errors: List[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_cube_state(state: List[int]) -> List[str]:
    """
    Validate a raw 54-element facelet array.
    Returns a list of error strings (empty = valid).
    """
    errors = []

    # Check value range
    if not all(0 <= v <= 5 for v in state):
        errors.append("All facelet values must be in range 0–5.")
        return errors

    # Check color count (each color must appear exactly 9 times)
    from collections import Counter
    counts = Counter(state)
    for color in range(6):
        if counts[color] != 9:
            color_names = ["U(white)", "L(orange)", "F(green)", "R(red)", "B(blue)", "D(yellow)"]
            errors.append(
                f"Color {color_names[color]} appears {counts[color]} times, expected 9."
            )

    # Check centers are correct (centers are fixed)
    expected_centers = {4: 0, 13: 1, 22: 2, 31: 3, 40: 4, 49: 5}
    for idx, color in expected_centers.items():
        if state[idx] != color:
            errors.append(f"Center at index {idx} must be color {color}, got {state[idx]}.")

    return errors


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "version": "0.1.0",
    }


@app.get("/info")
async def info():
    if _model is None:
        return {"model": "not loaded", "model_path": MODEL_PATH}
    return {
        "model": "F2L Solver",
        "model_path": MODEL_PATH,
        "input_shape": str(_model.input_shape),
        "output_shape": str(_model.output_shape),
        "parameters": int(_model.count_params()),
        "moves": MOVE_NAMES,
        "max_solve_moves": MAX_MOVES,
    }


@app.post("/validate", response_model=ValidateResponse)
async def validate(request: SolveRequest):
    errors = _validate_cube_state(request.state)
    if errors:
        return ValidateResponse(valid=False, cross_solved=False, f2l_already_solved=False, errors=errors)

    cube = CubeState(np.array(request.state, dtype=np.int8))
    return ValidateResponse(
        valid=True,
        cross_solved=is_cross_solved(cube),
        f2l_already_solved=is_f2l_solved(cube),
        errors=[],
    )


@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest):
    # ── 1. Validate ──────────────────────────────────────────────────────
    errors = _validate_cube_state(request.state)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    cube = CubeState(np.array(request.state, dtype=np.int8))
    progress = f2l_progress(cube)

    # ── 2. Guard conditions ──────────────────────────────────────────────
    if not is_cross_solved(cube):
        raise HTTPException(
            status_code=400,
            detail="Cross is not solved. NeuralCube assumes the cross is already complete."
        )

    if is_f2l_solved(cube):
        return SolveResponse(
            moves=[], move_count=0, solved=True,
            f2l_progress=progress,
            error=None,
        )

    # ── 3. Model check ───────────────────────────────────────────────────
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train a model first and place it at {MODEL_PATH}"
        )

    # ── 4. Inference ─────────────────────────────────────────────────────
    moves, solved = greedy_solve(_model, cube, max_moves=MAX_MOVES)
    final_progress = f2l_progress(cube.apply_moves(moves))

    return SolveResponse(
        moves=moves,
        move_count=len(moves),
        solved=solved,
        f2l_progress=final_progress,
        error=None if solved else f"Could not solve F2L within {MAX_MOVES} moves.",
    )
