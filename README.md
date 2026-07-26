# 🧩 NeuralCube

### An ML-Powered Hybrid Rubik's Cube F2L Solver

> **Status:** 🚧 Active Development  
> Backend Complete • Dataset Generated • Model Training In Progress • Frontend Planned

---

## Overview

NeuralCube is an experimental machine learning project that explores how neural networks can solve the **First Two Layers (F2L)** of a Rubik's Cube while remaining explainable, reliable, and efficient.

Unlike conventional cube solvers that rely entirely on deterministic search, NeuralCube combines three complementary solving strategies:

- 🧠 Neural Network inference
- 📘 Deterministic 41-case F2L solver
- ⚡ Kociemba two-phase algorithm

This cascading pipeline prioritizes learning-based reasoning while guaranteeing that every valid cube state receives a correct solution.

---

## Why Hybrid AI?

Traditional machine-learning cube solvers often struggle when they encounter unfamiliar cube states, become trapped in local minima, or accumulate prediction errors over long solving sequences.

NeuralCube addresses these limitations through a hybrid architecture:

- Neural Network for human-like pattern recognition
- Rule-Based Solver for deterministic handling of all canonical F2L cases
- Kociemba Search as a final guaranteed fallback

Rather than replacing classical algorithms, NeuralCube demonstrates how machine learning and algorithmic reasoning can complement one another to build more reliable intelligent systems.

---

## System Architecture

```
                User
                  │
                  ▼
           FastAPI REST API
                  │
                  ▼
      Cascading Solver Pipeline
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
 Neural Network  Rule Solver  Kociemba
     │            │            │
     └────────────┴────────────┘
                  │
                  ▼
         Solution Returned
```

The solver attempts each stage sequentially, only falling back when the previous stage cannot produce a valid solution.

---

## Machine Learning Pipeline

```
Solved Cube
      │
      ▼
Random Scramble Generator
      │
      ▼
Depth-Based Sampling
      │
      ▼
BFS / 2-Ply Label Generation
      │
      ▼
324-Dimensional Feature Encoding
      │
      ▼
Dense Neural Network
      │
      ▼
TensorFlow Model
```

The dataset is generated automatically using custom tooling rather than relying on existing cube databases.

---

## Features

- Hybrid AI solving pipeline
- Custom Rubik's Cube engine
- 200,000 automatically generated training samples
- Beam-search neural inference
- Deterministic 41-case F2L solver
- Kociemba fallback solver
- FastAPI REST backend
- Explainable solving pipeline
- TensorFlow.js deployment (planned)
- Interactive React + Three.js frontend (planned)

---

## Dataset

NeuralCube uses a custom-generated supervised learning dataset consisting of **200,000 labeled cube states**.

| Batch | Depth Range | Label Generation |
|---------|-------------|-----------------|
| Batch 1 | 1–4 | BFS |
| Batch 2 | 5–7 | 2-Ply Heuristic |
| Batch 3 | 8–10 | 2-Ply Heuristic |
| Batch 4 | 11–14 | 2-Ply Heuristic |

Features are encoded as **324-dimensional one-hot vectors**, representing all 54 facelets of the cube.

---

## Neural Network

Current architecture:

```
324
 │
 ▼
Dense (512)
 │
 ▼
Dense (256)
 │
 ▼
Dense (128)
 │
 ▼
Dense (64)
 │
 ▼
Softmax (18 Moves)
```

Training includes:

- Batch Normalization
- Dropout
- Early Stopping
- ReduceLROnPlateau
- Model Checkpointing

---

## REST API

Current endpoints:

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Model information |
| POST | `/solve` | Full hybrid solver |
| POST | `/solve/nn` | Neural Network only |
| POST | `/solve/rules` | Rule-based solver |
| POST | `/validate` | Cube validation |

---

## Repository Structure

```
.
├── neuralcube/
│   ├── cube/
│   ├── data/
│   ├── model/
│   ├── solver/
│   ├── api/
│   └── eval/
│
├── neuralcube-web/
│   └── React + Three.js frontend
│
├── data/
│   └── Generated datasets
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

### Machine Learning

- TensorFlow
- Keras
- NumPy

### Backend

- Python
- FastAPI
- Uvicorn

### Algorithms

- Beam Search
- Breadth-First Search (BFS)
- Kociemba Two-Phase Algorithm
- Rule-Based F2L Solver

### Frontend (Planned)

- React
- Three.js
- TensorFlow.js

---

## Development Progress

| Component | Status |
|------------|---------|
| Cube Engine | ✅ Complete |
| Dataset Generator | ✅ Complete |
| 200k Dataset | ✅ Complete |
| Neural Network Architecture | ✅ Complete |
| Training Pipeline | 🚧 In Progress |
| REST API | ✅ Complete |
| Evaluation Suite | ✅ Complete |
| React Frontend | 📝 Planned |
| TensorFlow.js Deployment | 📝 Planned |

---

## Roadmap

### Completed

- Cube engine
- F2L checker
- Dataset generation
- BFS labeler
- 2-Ply heuristic
- Rule-based F2L solver
- FastAPI backend

### Currently Working On

- Neural network training
- Hyperparameter tuning
- Model evaluation

### Planned

- React frontend
- Three.js visualization
- TensorFlow.js deployment
- Interactive solve animation
- Public web deployment

---

## Future Work

Future improvements include:

- Transformer-based cube reasoning
- Reinforcement learning experiments
- Model quantization for TensorFlow.js
- Mobile optimization
- Interactive benchmarking dashboard

---

## Code Availability

This project is currently under active development. The repository is publicly available to document its architecture, implementation progress, and engineering decisions.

The source code is provided for educational and portfolio purposes only. Reuse, redistribution, or commercial use is not permitted without prior written permission from the authors. A license will be added once the project reaches a stable release.

---

## Acknowledgements

NeuralCube is an independent portfolio project exploring the intersection of machine learning, algorithm design, and intelligent software systems.