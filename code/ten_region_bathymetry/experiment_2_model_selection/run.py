from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent; PARENT = HERE.parent
if str(PARENT) not in sys.path: sys.path.insert(0, str(PARENT))
from experiment_2_model_selection.workflow import load_selection_config, load_config, qc, report, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare regressors on identical track/spatial folds")
    ap.add_argument("command", choices=["qc", "run", "report"]); ap.add_argument("--main-run-id", required=True)
    ap.add_argument("--main-config", type=Path, default=PARENT / "experiment_1_main" / "config.yaml"); ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--max-tasks", type=int); args = ap.parse_args()
    main_cfg = load_config(args.main_config); cfg = load_selection_config(args.config)
    if args.command == "qc": result = qc(main_cfg, cfg, args.main_run_id)
    elif args.command == "run": result = run(main_cfg, cfg, args.main_run_id, args.max_tasks)
    else: result = report(main_cfg, cfg, args.main_run_id)
    print(result)


if __name__ == "__main__": main()
