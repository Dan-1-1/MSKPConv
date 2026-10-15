"""Backward-compatible import surface for Experiment 3."""
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_3_regional_adaptation import workflow as _implementation

globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})
