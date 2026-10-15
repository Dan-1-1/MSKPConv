from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_3_regional_adaptation.workflow import (  # noqa: E402
    load_adaptation_config,
    load_config,
    qc,
    report,
    run,
)


def _csv_list(value: str | None, cast):
    if not value:
        return None
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ten-region track-level regional adaptation experiment")
    parser.add_argument("command", choices=["qc", "run", "report"])
    parser.add_argument("--main-run-id", required=True)
    parser.add_argument("--main-config", type=Path, default=PARENT / "experiment_1_main" / "config.yaml")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--targets")
    parser.add_argument("--methods")
    parser.add_argument("--proportions")
    parser.add_argument("--seeds")
    args = parser.parse_args()
    main_cfg = load_config(args.main_config)
    adapt_cfg = load_adaptation_config(args.config)
    if args.command == "qc":
        result = qc(main_cfg, adapt_cfg, args.main_run_id)
    elif args.command == "report":
        result = report(main_cfg, adapt_cfg, args.main_run_id)
    else:
        result = run(main_cfg, adapt_cfg, args.main_run_id, args.max_tasks,
                     _csv_list(args.targets, str), _csv_list(args.methods, str),
                     _csv_list(args.proportions, float), _csv_list(args.seeds, int))
    print(result)


if __name__ == "__main__":
    main()
