# NeuralCube — F2L Backend

## Project Structure
```
neuralcube/
├── cube/
│   ├── __init__.py
│   ├── state.py          # Cube state representation + move logic
│   └── f2l_checker.py    # F2L solved checker
├── data/
│   ├── __init__.py
│   ├── scrambler.py      # Random scramble generator
│   ├── bfs_labeler.py    # BFS optimal-move labeler
│   └── generator.py      # Dataset builder → saves .npy files
├── model/
│   ├── __init__.py
│   ├── network.py        # Keras model definition
│   └── train.py          # Training script
├── eval/
│   ├── __init__.py
│   └── benchmark.py      # Solve rate + move efficiency
├── api/
│   ├── __init__.py
│   └── server.py         # FastAPI server
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Use Python 3.9, 3.10, or 3.11 for this project. TensorFlow does not provide Windows wheels for Python 3.14, so installing with the default `python` on this machine will fail unless it points to a supported interpreter.

## Quick Start

### 1. Generate training data
```bash
python -m data.generator --samples 200000 --max-depth 12 --out data/dataset
```

### 2. Train the model
```bash
python -m model.train --data data/dataset --epochs 50 --out model/saved/f2l_model.h5
```

### 3. Evaluate
```bash
python -m eval.benchmark --model model/saved/f2l_model.h5 --samples 10000
```

### 4. Run API server
```bash
uvicorn api.server:app --reload --port 8000
```

### API Usage
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"state": [0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, ...]}'
```
