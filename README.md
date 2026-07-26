# sNNake v5 — A Neural Network as a Game Engine

**The thesis:** when building a world model, separate what's learnable from
what's computable. Don't make a neural network rediscover integer arithmetic.

## What this project demonstrates

A 13.8K-parameter neural network that serves as a Snake game engine by only
learning what it needs to — the direction transition function — while everything
else is exact deterministic math.

| Version | Approach | Params | Result |
|---------|----------|--------|--------|
| v1–v4 | Neural net predicts *everything* (head, body, food, collisions) | 150K+ | Autoregressive drift, collapses |
| **v5** | Neural net predicts only direction; physics is exact tensor ops | **13.8K** | Zero drift, infinite stability |

The model learned `(direction + action_offset) % 4` from 30,000 training
examples. It's a 12-entry lookup table encoded in 1,200 parameters. The point
isn't that this is hard — it's that previous versions tried to learn *everything*
and failed.

## Architecture

```
Input: direction (4) + action (3)
        │
        ├── Direction MLP (1.2K params)
        │     Linear(7→32) → ReLU → Linear(32→4)
        │
        └── Deterministic physics (exact, not learned):
              new_head = old_head + offset[predicted_dir]
              collisions, body shift, food detection
```

The food spawn predictor (body encoder GRU + MLP, 12.6K params) exists in the
codebase but is not needed for gameplay — food spawns randomly on empty cells.

## What's in this repo

| Component | File | Purpose |
|-----------|------|---------|
| Model | `src/snnake/model_v5.py` | 13.8K-param StructuredWorldModel |
| Training | `src/snnake/train_v5.py` | Training loop (CUDA/MPS/CPU) |
| Data collection | `src/snnake/collector_v2.py` | Generates training data from ground-truth engine |
| Ground truth engine | `src/snnake/engine.py` | Reference Snake implementation |
| ONNX export | `export_onnx.py` | Exports direction MLP → 2.2 KB ONNX |
| Web demo | `web/index.html` | Browser demo — the ONNX model plays Snake autonomously |

## Quick Start

```bash
# Setup
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# Generate training data
python -m snnake.collector_v2 --episodes 5000 --output data/games_v3_combined.npz

# Train
python -m snnake.train_v5 --data data/balanced_dir.npz --epochs 30 --lr 1e-3

# Export ONNX
python export_onnx.py

# Web demo
cd web && python3 -m http.server 8765
# Open http://localhost:8765 — the model plays by itself
```

## Results

| Metric | Value |
|--------|-------|
| Learned parameters | 13,848 |
| Direction accuracy | 100% (12/12 cases) |
| ONNX model size | 2.2 KB |
| Training time | ~30s on M1 Mac (MPS) |

## License

MIT
