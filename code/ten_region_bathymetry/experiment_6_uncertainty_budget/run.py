from __future__ import annotations

import argparse
from pathlib import Path

from workflow import run_budget


def main() -> None:
    parser = argparse.ArgumentParser(description="Proposed-only CUDEM-conditional uncertainty budget")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--skip-tide-network", action="store_true")
    parser.add_argument("--skip-rasters", action="store_true")
    args = parser.parse_args()
    print(run_budget(args.config, args.run_id, args.skip_tide_network, args.skip_rasters))


if __name__ == "__main__":
    main()

