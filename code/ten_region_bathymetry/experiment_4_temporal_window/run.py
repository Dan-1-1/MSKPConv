from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_4_temporal_window.workflow import (  # noqa: E402
    load_config,
    load_experiment_config,
    prepare,
    qc,
    report,
    run,
)


def _ints(value: str | None) -> list[int] | None:
    return [int(x.strip()) for x in value.split(",") if x.strip()] if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: temporal-window and seasonal-bias analysis")
    parser.add_argument("command", choices=["qc", "prepare", "run", "report"])
    parser.add_argument("--main-run-id", required=True)
    parser.add_argument("--main-config", type=Path, default=PARENT / "experiment_1_main" / "config.yaml")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--windows", help="Optional comma-separated window subset")
    parser.add_argument("--folds", help="Optional comma-separated fold subset")
    parser.add_argument("--seeds", help="Optional comma-separated repeat-seed subset")
    args = parser.parse_args()
    main_cfg = load_config(args.main_config)
    cfg = load_experiment_config(args.config)
    if args.command == "qc":
        result = qc(main_cfg, cfg, args.main_run_id)
    elif args.command == "prepare":
        result = prepare(main_cfg, cfg, args.main_run_id)
    elif args.command == "report":
        result = report(main_cfg, cfg, args.main_run_id)
    else:
        result = run(main_cfg, cfg, args.main_run_id, args.max_tasks,
                     _ints(args.windows), _ints(args.folds), _ints(args.seeds))
    print(result)


if __name__ == "__main__":
    main()
