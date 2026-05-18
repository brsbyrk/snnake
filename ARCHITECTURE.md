# sNNake Architecture

## Design Decisions

### 1. Grid Encoding — 4-Channel One-Hot

**Decision:** Each cell is encoded as a 4-element one-hot vector across 4 channels.

| Channel | Meaning |
|---|---|
| 0 | Empty |
| 1 | Body segment |
| 2 | Head |
| 3 | Food |

**Rationale:** A single-channel grid with ordinal values (0,1,2,3) implies a false ordering — the network would learn that "2 > 1" matters, when really these are distinct categorical states. One-hot encoding removes this bias and lets the CNN learn spatial relationships from scratch.

**Alternative considered:** Single integer channel. Simpler, but introduces bias that hurts convergence.

### 2. Action Space — 3 Relative Actions

**Decision:** `[left, straight, right]` — relative to the snake's current heading.

**Rationale:** Prevents illegal reversals (180° turn into yourself) by construction. Simpler for the model to learn — 3 choices instead of 4 with one always invalid.

**Alternative considered:** 4 absolute directions (up/down/left/right). More natural for humans, but adds the constraint problem of invalid actions.

### 3. Model Architecture — CNN + MLP Hybrid

```
Input: grid (4, 10, 10)
       direction (4,)      one-hot
       action (3,)         one-hot
       game_over (1,)      binary

→ Conv2d(4→16, k=3, pad=1) + ReLU
→ Conv2d(16→32, k=3, pad=1) + ReLU
→ Flatten (32*10*10 = 3200)
→ Concat with [direction, action, game_over] → 3208
→ Linear(3208→256) + ReLU
→ Linear(256→256) + ReLU

Output heads:
  ─ Grid:     Linear(256→400) → reshape(4,10,10) → softmax per cell
  ─ Direction: Linear(256→4) → softmax
  ─ GameOver:  Linear(256→1) → sigmoid
```

**Rationale:**
- CNNs capture spatial structure (body shape, head-tail relationships, proximity to food)
- Small kernels (3×3) are sufficient for a 10×10 grid — a larger receptive field isn't needed
- Separate output heads allow heterogeneous loss functions
- 16→32 channels is sufficient for such small grids

**Alternative considered:** Pure MLP on flattened grid (400 → 128 → 128 → 405). Much worse spatial reasoning — the model can't learn translation invariance. Tested and confirmed poor convergence.

### 4. Loss Function — Weighted Multi-Task

```
loss = grid_loss + 0.5 * direction_loss + 0.5 * game_over_loss
```

- **Grid loss:** Cross-entropy per cell (4 classes), summed
- **Direction loss:** Cross-entropy over 4 directions
- **Game-over loss:** Binary cross-entropy

Lower weight on direction and game-over because they're easier to learn and the grid is the primary output.

### 5. Snake Rules

| Property | Value |
|---|---|
| Grid | 10 × 10 |
| Walls | Collision → game over (no wrapping) |
| Self-collision | Hit body → game over |
| Reversal | 180° turn prevented by action encoding |
| Growth | Eat food → +1 segment, tail stays, new food spawns |
| Initial position | Head at (5,5), direction right, length 1 |
| Food | Single piece, random free cell |
| Win condition | All cells filled → game over (victory) |

### 6. Data Collection Strategy

**Training data:** Run N games with random actions (direction changes with probability 0.1 per step) to maximize state diversity. Record `(state, action, next_state, done)` for every transition.

**Coverage:** ~50,000 episodes × ~20 steps/episode ≈ 1M transitions should cover the state space well. Random actions explore food positions, body configurations, and collision scenarios.

### 7. Autoregressive Validation

The true test: run the model in a closed loop — feed its own next-state prediction back as input, then predict again, iterate. If it drifts, the model has learned local transitions but not global dynamics.

Metrics:
- Trajectory accuracy (does it stay on valid trajectories for N steps?)
- Divergence rate (how many steps before illegal state or collapse?)
- Food tracking (does it correctly handle food consumption and respawn?)

### 8. Export

After training, export to ONNX for portable inference. The model becomes a ~500KB file that can run on CPU in <1ms per step — fast enough for real-time or embedded use.
