"""Backward-compatible CLI for Experiment 2."""
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_2_model_selection.run import main


if __name__ == "__main__":
    main()
