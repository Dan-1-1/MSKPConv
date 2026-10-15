"""Backward-compatible import surface for Experiment 1.

New code should import :mod:`experiment_1_main.workflow` directly.
"""
from experiment_1_main import workflow as _implementation

globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})
