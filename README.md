# sNNake — v5 Structured World Model

**A neural Snake game engine — exact physics with learned direction + food spawn.**

Instead of one big network trying to rediscover integer addition and shift registers, v5 separates the problem:

- **Deterministic physics (Exact arithmetic):** head movement, collision detection, body shift register, food eating — all computed as exact tensor ops. Zero drift, infinite autoregressive stability.
- **Learned components (<15K params):** direction update (direction + action → new direction) and food respawn (body context → new food cell).

## Architecture

```
Current state (head, direction, food, body, mask)
        │
        ├── Direction Net (1.2K params) 
        │     direction + action → new_dir_logits (4-class)
        │
        ├── Deterministic physics (exact):
        │     ├── new_head = old_head + dir_offset[new_dir]
        │     ├── wall_collision = out_of_bounds(new_head)
        │     ├── self_collision = new_head in body
        │     ├── ate = (new_head == food_pos)
        │     └── new_body = [new_head] + body[:-1]
        │
        └── Food Net (12.6K params)
              body_encoder(GRU) → MLP → food_logits (100-cell)
              (only used when ate=True)
```

## Results

| Metric | 15 epochs (5% noise) |
|--------|---------------------|
| Direction accuracy (val) | 99.8% |
| Learned params | 13,848 |
| Autoregressive mean | 127.1 steps |
| Autoregressive max | 1,307 steps |
| Samples ≥100 steps | 38.6% |
| Samples ≥500 steps | 5.4% |
| Samples ≥1,000 steps | 0.6% |
| AR test speed | 500 samples / 32s |

Because physics is deterministic, the model never "drifts" — it either predicts the correct direction or it doesn't. With direction accuracy at 99.8%, the expected consecutive correct steps is ~500. The first wrong direction prediction is a "death" — the trajectory ends, but doesn't compound into garbage.

## Quick Start

### Train
```bash
cd ~/workspace/_projects/snnake
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m snnake.train_v5 \
  --data data/games_v3_combined.npz \
  --epochs 15 --batch-size 256 --noise-prob 0.05
```

### Play in browser
```bash
cd ~/workspace/_projects/snnake/web
python3 -m http.server 8765
# Open http://localhost:8765 in a browser
```

The web player loads a **2.2 KB ONNX model** (direction MLP only) via ONNX Runtime Web. Deterministic physics runs in JavaScript. No server needed — pure client-side inference.

## Project Structure

```
snnake/
├── src/snnake/
│   ├── model_v5.py      # StructuredWorldModel (deterministic + learned)
│   ├── train_v5.py      # Training loop + AR evaluation
│   ├── model_v4.py      # Previous: discrete classification (deprecated)
│   ├── train_v4.py      # Previous trainer (deprecated)
│   ├── collector_v2.py  # Data collection pipeline
│   ├── encoding.py      # Direction/action encoding
│   └── engine.py        # Ground truth Snake game
├── web/                 # Browser playable game
│   ├── index.html       # Game page
│   ├── game.js          # Game logic + ONNX inference + render
│   ├── style.css        # Dark theme styling
│   └── direction_model.onnx  # 2.2 KB direction MLP
├── data/                # Training data (gitignored)
├── checkpoints_v5/      # v5 model weights (gitignored)
└── checkpoints_v4/      # v4 model weights (gitignored)
```

## Why v5?

Previous versions (v1-v4) tried to predict ALL game state from a neural network — head positions, body positions, collision flags — as classification or regression problems. This required 150K+ params and still suffered from autoregressive drift because any single wrong prediction cascades.

v5 recognizes that Snake physics are trivially computable. The only things worth learning are the direction transition (a deterministic `(dir + action_offset) % 4` that the network learns effortlessly) and food spawn distribution. This gives:

- **13.8K params** (vs 150K in v4)
- **Infinite AR stability** (exact arithmetic doesn't drift)
- **CPU-fast inference** (~1μs per step)
- **Near-perfect accuracy after 2 epochs**
