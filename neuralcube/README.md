# NeuralCube — F2L Backend

Neural network F2L solver with cascading fallback pipeline.

---

## Project Structure

```
neuralcube/
├── cube/
│   ├── state.py            # Cube state (54 facelets) + all 18 HTM moves
│   └── f2l_checker.py      # Cross + slot solved detection
├── data/
│   ├── scrambler.py        # Scramble generator (no curriculum bias)
│   ├── bfs_labeler.py      # BFS optimal-move labeler (depth limit = 6)
│   └── generator.py        # Batch generator with checkpointing + merge
├── model/
│   ├── network.py          # Keras model (Dense 512→256→128→64→18 softmax)
│   └── train.py            # Training script with callbacks
├── solver/
│   ├── nn_solver.py        # NN: beam search + cross-guard + loop detection
│   ├── f2l_case_solver.py  # Rule-based 41-case F2L solver (FR→FL→BR→BL)
│   ├── kociemba_solver.py  # Kociemba two-phase fallback
│   └── pipeline.py         # Master: Phase1 → Phase2 → Phase3
├── eval/
│   └── benchmark.py        # Solve rate + move count + phase breakdown
├── api/
│   └── server.py           # FastAPI: /solve /validate /solve/nn /solve/rules
└── requirements.txt
```

---

## Solver Pipeline

```
POST /solve
    │
    ▼
[Guard] Cross solved? → No → 400 error
[Guard] F2L already done? → Yes → return []
    │
    ▼
[Phase 1] Neural Network
    - Beam search (width 3)
    - Cross-guard: filters moves that break the cross
    - Loop detection: visited-states hash set
    - Move cap: 30
    ├─ Solved → return moves ✓
    │
    ▼
[Phase 2] Rule-Based 41-Case F2L Solver
    - Covers all 41 canonical F2L cases
    - Slot order: FR → FL → BR → BL
    - Deterministic, always terminates
    ├─ Solved → return moves ✓
    │
    ▼
[Phase 3] Kociemba Two-Phase Algorithm
    - Guaranteed solution in < 25 moves in milliseconds
    └─ return moves ✓
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Run Order

### Step 1 — Generate data in batches (run one per day or session)

Each batch covers a fixed depth range and saves independently.
Batches are safe to interrupt — they resume from the last checkpoint.

```bash
# Batch 1: depth 1–4  (easy states, ~10 min)
python -m data.generator --batch 1

# Batch 2: depth 5–7  (medium, ~20 min)
python -m data.generator --batch 2

# Batch 3: depth 8–10 (harder, ~30 min)
python -m data.generator --batch 3

# Batch 4: depth 11–14 (hardest, ~40 min)
python -m data.generator --batch 4
```

Check progress at any time:
```bash
python -m data.generator --status
```

If a batch is interrupted, just re-run the same command — it resumes automatically.

If you want to restart a batch from scratch:
```bash
python -m data.generator --batch 2 --no-resume
```

### Step 2 — Merge all completed batches

```bash
python -m data.generator --merge
# Output: data/dataset_X.npy  data/dataset_y.npy
```

Merge deduplicates and shuffles automatically.
You can merge partial batches (e.g. only batches 1+2 done) and train on those first.

### Step 3 — Train the model

```bash
python -m model.train --data data/dataset --epochs 50 --out model/saved/f2l_model.h5
```

### Step 4 — Benchmark

```bash
# Rule-based only (no model needed — run this first as a smoke test)
python -m eval.benchmark --rules-only --samples 500

# Full pipeline (requires trained model)
python -m eval.benchmark --model model/saved/f2l_model.h5 --samples 2000
```

### Step 5 — Start the API server

```bash
uvicorn api.server:app --reload --port 8000
```

---

## Batch Configuration

| Batch | Samples | Depth | Seed | Est. Time | BFS Label? |
|-------|---------|-------|------|-----------|------------|
| 1     | 50,000  | 1–4   | 1000 | ~10 min   | Always BFS (fast) |
| 2     | 50,000  | 5–7   | 2000 | ~20 min   | BFS + some heuristic |
| 3     | 50,000  | 8–10  | 3000 | ~30 min   | Mostly heuristic |
| 4     | 50,000  | 11–14 | 4000 | ~40 min   | All heuristic |
| **Total** | **200,000** | 1–14 | — | ~100 min | — |

**Why fixed depth ranges?** Curriculum mode (shallow→deep within one run) causes the batch to
slow down dramatically near the end as BFS struggles with deep states. Fixed ranges give
consistent speed throughout each run.

**Why BFS limit = 6?** At depth 8, BFS can explore millions of nodes per sample.
At depth 6, it stays fast while still producing optimal labels for most training states.
States deeper than 6 use a heuristic (greedy slot-maximizer) that is near-instant.

---

## File Outputs

```
data/
├── batches/
│   ├── batch_1_X.npy          ← Completed batch 1
│   ├── batch_1_y.npy
│   ├── batch_1_meta.json      ← Stats + config
│   ├── batch_2_ckpt_X.npy     ← In-progress checkpoint (batch 2)
│   ├── batch_2_ckpt_y.npy
│   ├── batch_2_ckpt_meta.json
│   └── ...
├── dataset_X.npy              ← Final merged + shuffled dataset
├── dataset_y.npy
└── dataset_meta.json
```

---

## API Response Format

```json
{
  "moves": ["R", "U", "R'", "U'"],
  "move_count": 4,
  "solved": true,
  "phase_used": "nn",
  "phase_detail": "solved",
  "time_ms": 12.4,
  "f2l_progress": {
    "cross_solved": true,
    "slots": {"FR": true, "FL": true, "BR": true, "BL": true},
    "slots_solved": 4,
    "f2l_complete": true
  }
}
```

---

## Color Encoding

| Int | Face | Color  |
|-----|------|--------|
| 0   | U    | White  |
| 1   | L    | Orange |
| 2   | F    | Green  |
| 3   | R    | Red    |
| 4   | B    | Blue   |
| 5   | D    | Yellow |

---

## Environment Variables (API server)

| Variable     | Default                    | Description           |
|--------------|----------------------------|-----------------------|
| `MODEL_PATH` | `model/saved/f2l_model.h5` | Path to trained model |
| `MAX_MOVES`  | `30`                       | NN inference move cap |
| `BEAM_WIDTH` | `3`                        | NN beam search width  |
