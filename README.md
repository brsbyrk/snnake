# sNNake

**A neural network that is the Snake game engine.**

Instead of hand-coded collision logic, movement rules, and growth mechanics, sNNake learns it all. Given the current game state and an action, a small CNN predicts the next game state — head position, body configuration, food, and game-over status.

The network *is* the environment. Run it autoregressively: feed its own output back as input and you get a fully neural Snake game.

## Quick Start

```bash
# Collect training data
uv run snnake-collect --episodes 50000 --output data/games.npz

# Train the world model
uv run snnake-train --data data/games.npz --epochs 50

# Watch it play (autoregressive model-as-engine mode)
uv run snnake-play --checkpoint checkpoints/best.pt
```

## Project Structure

```
snnake/
├── src/snnake/
│   ├── engine.py       # Ground truth Snake game (data source)
│   ├── encoding.py     # Grid/state encoding schemes
│   ├── model.py        # CNN world model architecture
│   ├── collector.py    # Data collection pipeline
│   ├── train.py        # Training loop + evaluation
│   └── play.py         # Autoregressive model playback
├── data/               # Training data (gitignored)
├── checkpoints/        # Saved model weights (gitignored)
├── docs/               # Design docs, experiments
├── ARCHITECTURE.md     # Full design rationale
└── pyproject.toml
```

## Why?

Most "Snake AI" projects train an agent to play the game well. sNNake trains a model to *simulate the game* — a world model approach. This is interesting because:

- **Compression** — a few hundred KB of weights replaces ~200 lines of game logic
- **Differentiable** — the game becomes a computation graph you can backprop through
- **Generalizable** — the same architecture can learn any grid-world dynamics

## Status

- [x] Ground truth engine
- [x] Grid encoding (4-channel one-hot)
- [x] CNN world model
- [x] Data collection pipeline
- [x] Training loop
- [x] Autoregressive playback
- [ ] Model actually trained and converged
