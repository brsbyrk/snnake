# sNNake v5 Architecture

## Design Philosophy

Snake physics are deterministic — head movement, collision detection, body shift
registers, and food-eating checks are exact arithmetic. The only things worth
learning are:

1. **Direction update:** `(old_direction + action_offset) % 4` — learned via a
   tiny neural network (1.2K params) so it's part of the model, not external logic.
2. **Food respawn:** where food appears after being eaten — learned via a
   body encoder GRU + MLP (12.6K params).

Total learned parameters: **13,848**. Everything else is exact tensor ops —
zero autoregressive drift.

## Architecture

```
Input: direction (4) + action (3)
       │
       ├── Direction MLP (1.2K params)
       │     Linear(7→32) → ReLU → Linear(32→4)
       │     Output: dir_logits (4-class)
       │
       └── Deterministic Physics (exact, in JS/Python):
             ├── new_head = old_head + dir_offset[predicted_dir]
             ├── wall_collision = out_of_bounds(new_head)
             ├── self_collision = new_head in body
             ├── ate = (new_head == food_pos)
             └── new_body = [new_head] + body[:-1]
                    (tail grows on food, shifts otherwise)

Food prediction (not used in web demo):
  body → GRU(embed=16, hidden=32) → context
  context → Linear(32→64) → ReLU → Linear(64→100)
  Output: food_logits (100-cell softmax)
```

## Key Decisions

### 1. Deterministic physics, learned direction

Previous versions (v1–v4) tried to predict ALL game state from a neural network —
head positions, body positions, collision flags — as classification or regression
problems. This required 150K+ params and still suffered from autoregressive drift.

v5 recognizes that physics are trivially computable. The direction network
learns `(dir + offset) % 4` from data, but the mapping is so simple that
accurate learning requires balanced training data (equal representation of all
12 direction-action pairs).

### 2. 3 relative actions

Action space: `[left, straight, right]` — relative to the snake's current heading.
Prevents illegal 180° reversals by construction. 3 choices instead of 4 with one
always invalid.

### 3. Direction encoding

4-element one-hot: `[N, E, S, W]`. Direction vectors: `(0,-1), (1,0), (0,1), (-1,0)`.

### 4. Grid

10×10 grid. Walls → game over. Self-collision → game over. Food → single piece,
random free cell.

## Export

After training, the direction MLP is exported to ONNX (2.2 KB). The web demo
loads it via ONNX Runtime Web and runs deterministic physics in JavaScript.
No server needed — pure client-side inference.

## Training Data

Generated via `collector.py` using the ground-truth `engine.py`. Balanced
direction data ensures the model learns turns, not just "go straight."
