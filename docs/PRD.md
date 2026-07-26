
  ## NEURALCUBE  
  F2L Neural Solver — Product Requirements Document  

Prepared by:  Rod
Role:  Computer Engineering Student · Mapúa University
Version:  0.2.0 (Backend Phase)
Date:  July 2026
Status:    IN DEVELOPMENT  

 
### 1. Executive Summary

NeuralCube is a browser-based Rubik's Cube F2L (First Two Layers) solver that uses a neural network as its primary reasoning engine, backed by a deterministic 41-case rule-based fallback and a Kociemba two-phase algorithm as a final guarantee. The project is a side project and portfolio showcase built by a Computer Engineering student specialising in Embedded Systems, demonstrating the convergence of machine learning, algorithm design, and edge AI deployment.

What Makes It Unique
▸  Mirrors human learning: neural net first, structured logic on failure, guaranteed algorithm as last resort
▸  Fully client-side inference via TensorFlow.js — no server round-trip during solve
▸  Transparent failure modes: every phase reports why it succeeded or handed off
▸  Graceful degradation pipeline not found in academic solvers or toy demos

Current Status (July 2026)
▸  Backend complete: cube engine, data pipeline, model architecture, rule-based solver, API server
▸  200,000 training samples generated across 4 depth-stratified batches
▸  Frontend: not yet started
 

### 2. Problem Statement

The Gap in Existing Solvers
Most AI Rubik's Cube solvers treat the puzzle as a pure optimisation problem — find the shortest path to solved, report moves, done. Tools like DeepCubeA, Kociemba, and AlphaZero-style solvers are technically impressive but share two common limitations:
▸  Black-box reasoning: no insight into why a move was chosen or where the solver struggled
▸  Server-dependent or desktop-only: not deployable as a lightweight, shareable browser tool

Failure Modes in Neural Cube Solvers
Standard imitation-learning solvers exhibit four documented failure patterns:

Failure Mode	What Happens	Frequency
Never Seen This Pattern	Model encounters structurally unfamiliar state → freezes on suboptimal move	Common at depth 10+
Local Minimum Trap	Individually good moves collectively loop (U U' U U' indefinitely)	~30% of failures
BFS Blind Spot	BFS labels a move that looks wrong locally but is globally optimal; model never learns it	Depth 10–15
Cumulative Error Spiral	92% per-move accuracy × 15 moves = 29% chance of perfect solve	All deep states

NeuralCube's Answer
NeuralCube addresses all four failure modes through its cascading pipeline: loop detection aborts infinite cycles early, cross-guard filters moves that would break the already-solved cross, and two deterministic fallback phases guarantee a correct solution is always returned.
 
### 3. Goals & Success Metrics

Primary Goals
▸  Train a neural network that solves F2L states in under 20 moves for 70%+ of depth 1–10 scrambles
▸  Implement a deterministic fallback that handles 100% of valid F2L states the NN misses
▸  Expose a clean REST API consumed by a future React/Three.js frontend
▸  Export the trained model to TensorFlow.js for client-side inference

Success Metrics
Metric	Target	Phase
NN solve rate (depth 1–10)	≥ 70%	Post-training
NN solve rate (depth 11–14)	≥ 40%	Post-training
Rule-based solve rate	100%	Current (verified)
Avg moves (NN, depth 1–10)	≤ 20 moves	Post-training
API response time	< 500 ms	Post-training
TF.js inference (browser)	< 100 ms	Frontend phase
Dataset size	200,000	Complete ✓
 
### 4. Tech Stack

Layer	Technology	Purpose
Cube Engine	Python / NumPy	54-facelet state, 18 HTM moves, F2L checker
Data Pipeline	Python / tqdm	Batch generator, BFS labeler, 2-ply heuristic, checkpointing
Model	TensorFlow / Keras	Dense 512→256→128→64→18 softmax, trained on 200k samples
Rule-Based Solver	Pure Python	Complete 41-case F2L solver, slot order FR→FL→BR→BL
Fallback	kociemba (Python lib)	Two-phase algorithm, guaranteed <25 moves in milliseconds
API Server	FastAPI / Uvicorn	REST endpoints: /solve /validate /solve/nn /solve/rules
Frontend (TBD)	React + Three.js	3D cube visualiser, move animation, solve UI
Edge Inference	TensorFlow.js	Client-side NN inference, no server round-trip
Dev Environment	VS Code / Windows PS	PowerShell-compatible tooling, no bash dependency

Key Design Decisions
▸  BFS depth limit set to 6 (not 8) — prevents exponential slowdown; depth 7+ uses 2-ply heuristic
▸  Four depth-stratified batches (1–4, 5–7, 8–10, 11–14) for consistent generation speed
▸  Fixed seeds per batch (1000/2000/3000/4000) guarantee zero sample overlap across runs
▸  Checkpoint every 5,000 samples — crash-safe; resume without restarting batch
▸  Beam search width 3 in NN inference — better than greedy, cheaper than full BFS
 
### 5. System Architecture
```
Solver Pipeline
Requests enter POST /solve and cascade through three phases:

POST /solve
    │
    ▼
[Guard] Cross solved? → No → 400 error
[Guard] F2L already done? → Yes → return []
    │
    ▼
[Phase 1] Neural Network
    · Beam search (width 3)
    · Cross-guard: skip moves that break cross facelets
    · Loop detection: visited-states hash set
    · Move cap: 30
    ├─ Solved → return moves ✓
    │
    ▼
[Phase 2] Rule-Based 41-Case F2L Solver
    · All 41 canonical cases covered
    · Slot order: FR → FL → BR → BL
    · Deterministic, always terminates
    ├─ Solved → return moves ✓
    │
    ▼
[Phase 3] Kociemba Two-Phase Algorithm
    · Guaranteed solution in < 25 moves
    · Runs in milliseconds
    └─ return moves ✓

Data Generation Pipeline
Solved cube → Random scramble (N moves) → BFS label (depth ≤6) / 2-ply heuristic (depth 7+)
→ Encode state (54 facelets × 6 one-hot = 324-dim vector) → (X, y) pair → .npy file

API Endpoints
Method	Endpoint	Description
GET	/	Health check — model loaded status, version
GET	/info	Pipeline config, move list, beam width, move cap
POST	/solve	Full cascading pipeline — NN → rules → Kociemba
POST	/validate	Validate cube state without solving
POST	/solve/nn	NN phase only — for benchmarking model in isolation
POST	/solve/rules	Rule-based only — deterministic, no model required
```
### 6. Data Pipeline
```
Batch Configuration
Batch	Samples	Depth Range	Seed	BFS Limit	Status
1	50,000	1–4	1000	6	✓ Complete
2	50,000	5–7	2000	0 (heuristic)	✓ Complete
3	50,000	8–10	3000	0 (heuristic)	✓ Complete
4	50,000	11–14	4000	0 (heuristic)	✓ Complete
```
Labeling Strategy
Depth	Labeler	Quality	Speed
1–6	BFS (limit 6)	Optimal — guaranteed shortest path	Fast (few million nodes)
7–14	2-ply heuristic	Near-optimal — best 2-step lookahead	Near-instant (324 calls)

2-ply Heuristic (Option C)
The 2-ply heuristic evaluates all 18×18 = 324 (move1, move2) pairs and returns the move1 whose 2-step outcome has the highest composite score. This correctly identifies setup moves — moves that appear neutral at ply-1 but unlock a much better state at ply-2. Score = slots_solved × 1000 + cross_intact × 500 + correct_facelets × 1.
 
### 7. Project Folder Structure

Visual directory tree (ASCII) with descriptions of what each folder contains:
```
neuralcube/
├── cube/                   ← Core cube physics layer
│   ├── __init__.py
│   ├── state.py            ← CubeState class, 54-facelet array, all 18 HTM moves,
│   │                          one-hot encoder, move application logic
│   └── f2l_checker.py      ← Detects if cross / individual slots / full F2L
│                              are solved; returns progress dict for API response
│
├── data/                   ← Dataset generation layer
│   ├── __init__.py
│   ├── scrambler.py        ← Applies N random HTM moves to a solved cube;
│   │                          avoids same-face repeats and opposite-face triples
│   ├── bfs_labeler.py      ← BFS (depth ≤ 6) or 2-ply heuristic labeler;
│   │                          returns the optimal next move index (0–17)
│   └── generator.py        ← Batch manager: fixed depth ranges per batch,
│                              checkpoint every 5k samples, resume on crash,
│                              --status / --merge / --batch N CLI interface
│
├── data/batches/           ← Generated sample files (gitignored, large)
│   ├── batch_1_X.npy       ← Float32 array (50000, 324) — encoded cube states
│   ├── batch_1_y.npy       ← Int32  array (50000,)      — move labels (0–17)
│   ├── batch_1_meta.json   ← Stats: samples, depth range, seed, elapsed time
│   ├── batch_N_ckpt_X.npy  ← In-progress checkpoint (overwritten every 5k)
│   └── ...
│
├── data/dataset_X.npy      ← Final merged + deduplicated + shuffled dataset
├── data/dataset_y.npy
├── data/dataset_meta.json
│
├── model/                  ← Neural network layer
│   ├── __init__.py
│   ├── network.py          ← Keras model: Dense(512→256→128→64→18 softmax),
│   │                          BatchNorm + Dropout between layers, build/load helpers
│   ├── train.py            ← Training pipeline: load data, split, fit with
│   │                          EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
│   ├── logs/               ← TensorBoard training logs (auto-generated)
│   └── saved/
│       └── f2l_model.h5    ← Best checkpoint saved by ModelCheckpoint callback
│
├── solver/                 ← Three-phase solver layer
│   ├── __init__.py
│   ├── nn_solver.py        ← Phase 1: beam search (width 3), cross-guard,
│   │                          visited-state loop detection, safe move filter
│   ├── f2l_case_solver.py  ← Phase 2: all 41 F2L cases, slot order FR→FL→BR→BL,
│   │                          U-rotation trick to reuse FR algorithms for all slots
│   ├── kociemba_solver.py  ← Phase 3: wraps kociemba library, converts facelet
│   │                          array to kociemba string format, normalises output
│   └── pipeline.py         ← Master orchestrator: runs phases in order, preserves
│                              partial NN progress, returns unified result dict
│
├── eval/                   ← Evaluation & benchmarking layer
│   ├── __init__.py
│   └── benchmark.py        ← Tests each phase independently; reports solve rate,
│                              avg moves, avg time, breakdown by depth bucket
│
├── api/                    ← REST API layer
│   ├── __init__.py
│   └── server.py           ← FastAPI app: /solve /validate /solve/nn /solve/rules,
│                              loads model at startup, state validator, CORS enabled
│
├── requirements.txt        ← Python dependencies: numpy tensorflow fastapi
│                              uvicorn pydantic tqdm kociemba
└── README.md               ← Setup, batch run order, API docs, color encoding
```
Folder Responsibilities at a Glance
Folder	Contains	Can run without…
cube/	Physics — state representation and move logic	Everything else
data/	Generation — scrambles, labels, batch management	model/, solver/, api/
model/	Learning — Keras architecture and training loop	solver/, api/
solver/	Intelligence — NN + rules + Kociemba + pipeline	api/ (call directly)
eval/	Measurement — solve rate and efficiency benchmarks	api/
api/	Interface — FastAPI REST server for frontend consumption	Nothing (top-level)
 
### 8. User Experience

Target Users
▸  Rubik's Cube enthusiasts who want to understand F2L, not just get a solution
▸  Students and developers curious about how neural networks reason about puzzles
▸  Portfolio viewers evaluating the author's ML and systems engineering depth

Core User Flow (Planned Frontend)
1.  User arrives at the browser app — a 3D interactive Rubik's Cube is displayed.
2.  User inputs their cube state (color picker per facelet or notation entry).
3.  App validates: cross must be solved before F2L can proceed.
4.  User taps Solve — the app calls POST /solve on the FastAPI backend.
5.  Response arrives with the move sequence and which phase solved it.
6.  The 3D cube animates each move step-by-step at adjustable speed.
7.  A sidebar shows which F2L slot was solved at each phase of the solution.
8.  A phase indicator shows: NN / Rules / Kociemba — making the solver's reasoning transparent.

UX Principles
▸  Transparency over magic — always show which solver phase was used and why
▸  Progressive disclosure — basic mode shows just moves; advanced mode shows slot-by-slot breakdown
▸  Edge-first — TF.js inference runs in the browser; API fallback only if model fails to load
▸  Mobile-friendly — 3D cube must be usable on touch devices
```
API Response to Frontend
{
  "moves":        ["R", "U", "R'", "U'", "F'", "U", "F"],
  "move_count":   7,
  "solved":       true,
  "phase_used":   "nn",
  "phase_detail": "solved",
  "time_ms":      14.2,
  "f2l_progress": {
    "cross_solved":  true,
    "slots":         { "FR": true, "FL": true, "BR": true, "BL": true },
    "slots_solved":  4,
    "f2l_complete":  true
  }
}
```
### 9. Progress Tracker

Component	Task	Status
Cube Engine	54-facelet state, 18 HTM moves, one-hot encoder	✓ Done
Cube Engine	F2L checker (cross + 4 slots)	✓ Done
Data Pipeline	Scramble generator (no same-face repeats)	✓ Done
Data Pipeline	BFS labeler (depth limit 6)	✓ Done
Data Pipeline	2-ply heuristic labeler (Option C)	✓ Done
Data Pipeline	Batch generation (4 batches × 50k, checkpointing)	✓ Done
Data Pipeline	200k samples generated across all 4 batches	✓ Done
Data Pipeline	Merge + deduplicate + shuffle → dataset_X/y.npy	⟳ Next
Model	Keras architecture (Dense 512→256→128→64→18)	✓ Done
Model	Training script with callbacks	✓ Done
Model	Train on 200k dataset	⟳ Next
Solver	NN solver: beam search + cross-guard + loop detection	✓ Done
Solver	Rule-based 41-case F2L solver	✓ Done
Solver	Kociemba fallback wrapper	✓ Done
Solver	Cascading pipeline orchestrator	✓ Done
API	FastAPI server + all endpoints	✓ Done
Eval	Benchmark: rules-only + full pipeline	✓ Done
Frontend	React + Three.js 3D cube visualiser	— Planned
Frontend	TF.js model export + client-side inference	— Planned
Frontend	Move animation + slot-by-slot breakdown	— Planned
 
### 10. Risks & Mitigations

Risk	Likelihood	Impact	Mitigation
NN solve rate below 70% after training	Medium	Medium	Rule-based fallback guarantees 100% solve rate regardless; NN is an enhancement not a requirement
Training too slow on local CPU	High	Low	Use Google Colab free GPU tier; model is small (~500k params), trains in <1 hour on GPU
kociemba library not available on target platform	Low	Low	Phase 2 (rule-based) already guarantees 100% solve; Kociemba is belt-and-suspenders
TF.js model too large for fast browser load	Medium	Medium	Apply post-training quantisation; target < 5MB; lazy-load model after page interactive
41-case solver misses an edge case	Low	Medium	Kociemba as Phase 3 covers any gap; benchmark --rules-only catches regressions

