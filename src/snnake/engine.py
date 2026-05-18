"""Ground truth Snake game engine.

Standalone Snake simulation with full collision, growth, and food logic.
Used as the data source for training the world model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .encoding import (
    GRID_SIZE,
    EMPTY, BODY, HEAD, FOOD,
    encode_state,
    encode_action,
    apply_action,
    direction_from_index,
    index_from_direction,
    DIRECTION_VECTORS,
)

GRID = GRID_SIZE  # alias for readability


@dataclass
class SnakeState:
    """Full game state."""
    body: List[Tuple[int, int]]  # (x, y) from head to tail
    direction_idx: int           # 0=up, 1=right, 2=down, 3=left
    food_pos: Tuple[int, int]
    game_over: bool
    score: int
    grid: np.ndarray = field(init=False)  # (GRID, GRID) integer array

    def __post_init__(self):
        self._build_grid()

    def _build_grid(self):
        """Build integer grid from current state."""
        g = np.full((GRID, GRID), EMPTY, dtype=np.int32)
        # Head
        g[self.body[0][1], self.body[0][0]] = HEAD
        # Body (rest of segments)
        for x, y in self.body[1:]:
            g[y, x] = BODY
        # Food
        fx, fy = self.food_pos
        g[fy, fx] = FOOD
        self.grid = g


class SnakeEngine:
    """Ground truth Snake game engine.

    Usage:
        engine = SnakeEngine()
        state = engine.reset()
        state, action_idx, next_state, reward, done = engine.step(action_idx)
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._state: SnakeState | None = None

    def reset(self) -> SnakeState:
        """Reset the game to initial state."""
        # Start head at center, length 1
        start_x, start_y = GRID // 2, GRID // 2
        body = [(start_x, start_y)]

        # Direction: right
        direction_idx = 1  # right

        # Place food on a free cell
        food_pos = self._random_free_cell(body)

        self._state = SnakeState(
            body=list(body),
            direction_idx=direction_idx,
            food_pos=food_pos,
            game_over=False,
            score=0,
        )
        return self._state

    def step(self, action_idx: int) -> Tuple[SnakeState, int, SnakeState, float, bool]:
        """Apply an action and advance the game by one tick.

        Args:
            action_idx: 0=left, 1=straight, 2=right

        Returns:
            (prev_state, action_idx, next_state, reward, done)
            reward: +1 for eating food, 0 otherwise
            done: True if game is over
        """
        assert self._state is not None, "Call reset() first"
        assert not self._state.game_over, "Game is already over"
        assert 0 <= action_idx <= 2, f"Invalid action: {action_idx}"

        prev_state = self._state
        s = self._state

        # Compute new direction
        new_dir_idx = apply_action(s.direction_idx, action_idx)

        # Current head position
        head_x, head_y = s.body[0]
        dx, dy = direction_from_index(new_dir_idx)
        new_head = (head_x + dx, head_y + dy)
        nx, ny = new_head

        # --- Collision detection ---
        # Wall collision
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            s.game_over = True
            self._update_grid()
            return prev_state, action_idx, s, 0.0, True

        # Self collision (check against current body, including tail
        # which will move unless food is eaten)
        if new_head in s.body:
            s.game_over = True
            self._update_grid()
            return prev_state, action_idx, s, 0.0, True

        # --- Move ---
        s.body.insert(0, new_head)

        # Check food
        ate = new_head == s.food_pos
        if ate:
            s.score += 1
            # Don't pop tail → snake grows
            # Place new food
            free = self._find_free_cells()
            if free:
                s.food_pos = self.rng.choice(free)
            else:
                # Board full! Win condition
                s.game_over = True
                self._update_grid()
                return prev_state, action_idx, s, 1.0, True
        else:
            s.body.pop()  # remove tail

        s.direction_idx = new_dir_idx
        self._update_grid()

        done = s.game_over
        reward = 1.0 if ate else 0.0

        return prev_state, action_idx, s, reward, done

    def _update_grid(self):
        """Rebuild the grid array from current state."""
        self._state._build_grid()

    def _random_free_cell(self, body: list | None = None) -> Tuple[int, int]:
        """Get a random free cell not occupied by the snake body."""
        if body is None:
            body = self._state.body
        occupied = set(body)
        free = [(x, y) for x in range(GRID) for y in range(GRID) if (x, y) not in occupied]
        return self.rng.choice(free)

    def _find_free_cells(self) -> list[Tuple[int, int]]:
        """Return all cells not occupied by snake body."""
        occupied = set(self._state.body)
        return [(x, y) for x in range(GRID) for y in range(GRID) if (x, y) not in occupied]

    def _dir_vector(self, state: SnakeState | None = None) -> tuple:
        """Get direction vector from state."""
        if state is None:
            state = self._state
        return DIRECTION_VECTORS[state.direction_idx]

    @property
    def state(self) -> SnakeState | None:
        return self._state

    def encode_state(self, state: SnakeState | None = None) -> tuple:
        """Encode a state into network inputs.

        Returns (grid_tensor, dir_tensor, go_tensor) as numpy arrays.
        """
        if state is None:
            state = self._state
        return encode_state(state.grid, DIRECTION_VECTORS[state.direction_idx], state.game_over)

    @staticmethod
    def state_to_observation(state: SnakeState) -> dict:
        """Human-readable observation."""
        return {
            "head": state.body[0],
            "body": state.body,
            "length": len(state.body),
            "direction": DIRECTION_VECTORS[state.direction_idx],
            "food": state.food_pos,
            "score": state.score,
            "game_over": state.game_over,
        }
