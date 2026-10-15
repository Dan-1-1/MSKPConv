from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from scene_evaluation.workflow import load_scene_config, qc, report, run  # noqa: E402
from experiment_1_main.workflow import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Representative-scene CUDEM validation")
    parser.add_argument("command", choices=["qc", "run", "report"])
    parser.add_argument("--main-config", type=Path, default=Path(__file__).parents[1] / "experiment_1_main" / "config.yaml")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--main-run-id", required=True)
    parser.add_argument("--max-regions", type=int, default=None)
    args = parser.parse_args()

    main_cfg = load_config(args.main_config)
    scene_cfg = load_scene_config(args.config)
    if args.command == "qc":
        print(qc(main_cfg, scene_cfg, args.main_run_id))
    elif args.command == "run":
        print(run(main_cfg, scene_cfg, args.main_run_id, args.max_regions))
    else:
        print(report(main_cfg, scene_cfg, args.main_run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
