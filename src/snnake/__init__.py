"""sNNake — A neural network that is the Snake game engine."""

from .engine import SnakeEngine
from .encoding import encode_state, decode_grid, state_to_grid
from .model import WorldModel
from .collector import collect_data
from .train import train

__version__ = "0.1.0"
