from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_1_main.workflow import load_config, qc, prepare, split, train, train_final_convex_blend, predict, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Ten-region photon classification bathymetry experiment")
    parser.add_argument("command", choices=["qc", "prepare", "split", "train", "train-final", "predict", "report"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--run-id", default=None, help="Reuse the same output directory across stages")
    parser.add_argument("--max-tasks", type=int, default=None, help="Train at most N fold tasks (smoke test)")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--regions", nargs="*", default=None, help="Prepare only these configured region IDs")
    parser.add_argument("--force-methods", nargs="*", default=None, help="Rebuild these prepared methods even when checkpoints exist")
    parser.add_argument(
        "--track-selection-csv",
        type=Path,
        default=None,
        help="Optional manifest; only rows with selected=True are used for matching",
    )
    parser.add_argument(
        "--scene-match-days",
        type=int,
        default=None,
        help="Maximum track-to-Sentinel-2 acquisition gap for matching",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.track_selection_csv is not None:
        cfg["matching_track_selection_csv"] = str(args.track_selection_csv.resolve())
    if args.scene_match_days is not None:
        if args.scene_match_days <= 0:
            raise SystemExit("--scene-match-days must be positive")
        cfg["scene_match_days"] = int(args.scene_match_days)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"run_id={run_id}")
    if args.command == "qc": qc(cfg, run_id)
    elif args.command == "prepare":
        prepare(
            cfg,
            run_id,
            set(args.regions) if args.regions else None,
            set(args.force_methods) if args.force_methods else None,
        )
    elif args.command == "split": split(cfg, run_id)
    elif args.command == "train": train(cfg, run_id, args.max_tasks, args.folds)
    elif args.command == "train-final": print(train_final_convex_blend(cfg, run_id))
    elif args.command == "predict": predict(cfg, run_id)
    elif args.command == "report": report(cfg, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
