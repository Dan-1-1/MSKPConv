"""Backward-compatible import surface for Experiment 2."""
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_2_model_selection import workflow as _implementation

globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})
