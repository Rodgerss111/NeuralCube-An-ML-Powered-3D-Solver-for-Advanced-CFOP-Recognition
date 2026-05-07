"""
api/server.py
-------------
FastAPI REST API for the NeuralCube F2L solver.

Uses the cascading fallback pipeline:
  Phase 1 → Neural network (beam search, cross-guard, loop detection)
  Phase 2 → Rule-based 41-case F2L solver (deterministic, always correct)
  Phase 3 → Kociemba two-phase algorithm (guaranteed <25 moves)

Endpoints:
  GET  /            — health check
  GET  /info        — model + pipeline info
  POST /solve       — full pipeline
  POST /validate    — validate cube state only
  POST /solve/nn    — NN phase only (for benchmarking)
  POST /solve/rules — rule-based only (no NN, no Kociemba)

Run:
  uvicorn api.server:app --reload --port 8000

Color encoding (state array):
  0=U(white) 1=L(orange) 2=F(green) 3=R(red) 4=B(blue) 5=D(yellow)
"""

import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from typing import Optional
from collections import Counter

from cube.state import CubeState, MOVE_NAMES
from cube.f2l_checker import is_f2l_solved, is_cross_solved, f2l_progress
from solver.pipeline import solve
from solver.f2l_case_solver import solve_f2l
from solver.nn_solver import nn_solve

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH",   "model/saved/f2l_model.h5")
MAX_MOVES  = int(os.environ.get("MAX_MOVES",  "30"))
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "3"))

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if os.path.exists(MODEL_PATH):
        from model.network import load_model
        print(f"Loading model from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH)
        print("Model loaded.")
    else:
        print(
            f"No model at {MODEL_PATH}. "
            "NN phase will be skipped; rule-based + Kociemba remain active."
        )
    yield


app = FastAPI(
    title="NeuralCube F2L Solver",
    description="Cascading F2L solver: NN → 41-case rules → Kociemba",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    state: list[int]

    @field_validator("state")
    @classmethod
    def check_length(cls, v):
        if len(v) != 54:
            raise ValueError(f"State must have 54 facelets, got {len(v)}")
        return v


class SolveResponse(BaseModel):
    moves: list[str]
    move_count: int
    solved: bool
    phase_used: str
    phase_detail: str
    time_ms: float
    f2l_progress: dict
    error: Optional[str] = None


class ValidateResponse(BaseModel):
    valid: bool
    cross_solved: bool
    f2l_already_solved: bool
    errors: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

_FACE_NAMES = ["U(white)", "L(orange)", "F(green)", "R(red)", "B(blue)", "D(yellow)"]
_EXPECTED_CENTERS = {4: 0, 13: 1, 22: 2, 31: 3, 40: 4, 49: 5}


def _validate(state: list[int]) -> list[str]:
    errors = []
    if not all(0 <= v <= 5 for v in state):
        errors.append("All facelet values must be in range 0–5.")
        return errors
    counts = Counter(state)
    for c in range(6):
        if counts[c] != 9:
            errors.append(f"{_FACE_NAMES[c]} appears {counts[c]} times, expected 9.")
    for idx, color in _EXPECTED_CENTERS.items():
        if state[idx] != color:
            errors.append(f"Center at index {idx} must be color {color}, got {state[idx]}.")
    return errors


def _make_cube(state: list[int]) -> CubeState:
    return CubeState(np.array(state, dtype=np.int8))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "version": "0.2.0",
        "pipeline": ["nn", "case_solver", "kociemba"],
    }


@app.get("/info")
async def info():
    return {
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "pipeline": {
            "phase_1": "Neural network — beam search, cross-guard, loop detection",
            "phase_2": "Rule-based 41-case F2L solver (slot order: FR→FL→BR→BL)",
            "phase_3": "Kociemba two-phase algorithm (<25 moves guaranteed)",
        },
        "moves": MOVE_NAMES,
        "nn_max_moves": MAX_MOVES,
        "nn_beam_width": BEAM_WIDTH,
    }


@app.post("/validate", response_model=ValidateResponse)
async def validate_endpoint(req: SolveRequest):
    errors = _validate(req.state)
    if errors:
        return ValidateResponse(
            valid=False, cross_solved=False, f2l_already_solved=False, errors=errors
        )
    cube = _make_cube(req.state)
    return ValidateResponse(
        valid=True,
        cross_solved=is_cross_solved(cube),
        f2l_already_solved=is_f2l_solved(cube),
        errors=[],
    )


@app.post("/solve", response_model=SolveResponse)
async def solve_endpoint(req: SolveRequest):
    errors = _validate(req.state)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    cube = _make_cube(req.state)
    if not is_cross_solved(cube):
        raise HTTPException(
            status_code=400,
            detail="Cross is not solved. The solver requires a completed cross before F2L."
        )

    result = solve(
        cube,
        model=_model,
        use_nn=(_model is not None),
        use_case_solver=True,
        use_kociemba=True,
        nn_max_moves=MAX_MOVES,
        nn_beam_width=BEAM_WIDTH,
        verbose=False,
    )

    return SolveResponse(
        moves=result["moves"],
        move_count=result["move_count"],
        solved=result["solved"],
        phase_used=result["phase_used"],
        phase_detail=result["phase_detail"],
        time_ms=result["time_ms"],
        f2l_progress=result["f2l_progress"],
        error=None if result["solved"] else "All solver phases exhausted.",
    )


@app.post("/solve/nn", response_model=SolveResponse)
async def solve_nn_only(req: SolveRequest):
    """NN phase only — for benchmarking model performance in isolation."""
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded from {MODEL_PATH}.")
    errors = _validate(req.state)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    cube = _make_cube(req.state)
    if not is_cross_solved(cube):
        raise HTTPException(status_code=400, detail="Cross not solved.")

    t0 = time.perf_counter()
    moves, solved, reason = nn_solve(
        _model, cube, max_moves=MAX_MOVES, beam_width=BEAM_WIDTH
    )
    elapsed = (time.perf_counter() - t0) * 1000
    final = cube.apply_moves(moves)

    return SolveResponse(
        moves=moves, move_count=len(moves), solved=solved,
        phase_used="nn", phase_detail=reason,
        time_ms=round(elapsed, 2),
        f2l_progress=f2l_progress(final),
        error=None if solved else reason,
    )


@app.post("/solve/rules", response_model=SolveResponse)
async def solve_rules_only(req: SolveRequest):
    """Rule-based 41-case solver only — deterministic, no model required."""
    errors = _validate(req.state)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    cube = _make_cube(req.state)
    if not is_cross_solved(cube):
        raise HTTPException(status_code=400, detail="Cross not solved.")

    t0 = time.perf_counter()
    moves, solved = solve_f2l(cube)
    elapsed = (time.perf_counter() - t0) * 1000
    final = cube.apply_moves(moves)

    return SolveResponse(
        moves=moves, move_count=len(moves), solved=solved,
        phase_used="case_solver", phase_detail="41_cases",
        time_ms=round(elapsed, 2),
        f2l_progress=f2l_progress(final),
        error=None if solved else "Rule-based solver incomplete — check cube state validity.",
    )
