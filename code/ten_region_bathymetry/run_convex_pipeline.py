"""Run the remaining ConvexBlend experiments sequentially in a background-safe process."""
from __future__ import annotations

import json
import argparse
import shutil
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RUN_ID = "20261007T120000Z_full_depth"


def seed_prepared_regions(output_root: Path, source_run_id: str, run_id: str,
                          rebuild_region: str, configured_regions: set[str]) -> None:
    """Copy immutable prepared inputs for unchanged regions into a new run."""
    source = output_root / source_run_id / "prepared"
    destination = output_root / run_id / "prepared"
    if not source.is_dir():
        raise SystemExit(f"prepared source does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, object] = {}
    for region_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if region_dir.name == rebuild_region or region_dir.name not in configured_regions:
            continue
        target = destination / region_dir.name
        if target.exists():
            raise SystemExit(f"refusing to overwrite seeded prepared region: {target}")
        shutil.copytree(region_dir, target)
        summary = target / "summary.json"
        if summary.exists():
            copied[region_dir.name] = json.loads(summary.read_text(encoding="utf-8"))
    (destination / "summary.json").write_text(
        json.dumps(copied, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lineage = {
        "source_run_id": source_run_id,
        "target_run_id": run_id,
        "copied_regions": sorted(copied),
        "rebuilt_region": rebuild_region,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "reuse_lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_status(status_path: Path, run_id: str, stage: str, state: str,
                 returncode: int | None = None) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": stage,
                "state": state,
                "returncode": returncode,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_stage(status_path: Path, run_id: str, stage: str, command: list[str]) -> None:
    print(f"[pipeline] start {stage}: {' '.join(command)}", flush=True)
    write_status(status_path, run_id, stage, "running")
    completed = subprocess.run(command, cwd=REPO, check=False)
    if completed.returncode != 0:
        write_status(status_path, run_id, stage, "failed", completed.returncode)
        raise SystemExit(completed.returncode)
    write_status(status_path, run_id, stage, "completed", completed.returncode)
    print(f"[pipeline] completed {stage}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ordered ten-region experiment pipeline")
    parser.add_argument("--start-at", default=None, help="Start at a named stage while reusing completed outputs")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run-specific output/checkpoint identifier")
    parser.add_argument("--prepared-source-run-id", default=None,
                        help="Copy unchanged prepared regions from this completed run")
    parser.add_argument("--rebuild-region", default=None,
                        help="Only rematch this region after seeding the other prepared regions")
    parser.add_argument("--adaptation-targets", default=None,
                        help="Comma-separated regional-adaptation targets to rerun")
    parser.add_argument(
        "--main-config",
        type=Path,
        default=HERE / "experiment_1_main" / "config.yaml",
        help="Main experiment config; can be paired with --track-selection-csv",
    )
    parser.add_argument(
        "--track-selection-csv",
        type=Path,
        default=None,
        help="Use only selected=True tracks from this manifest during matching",
    )
    parser.add_argument(
        "--scene-match-days",
        type=int,
        default=None,
        help="Maximum track-to-scene gap; use 180 for the configured temporal windows",
    )
    args = parser.parse_args()
    run_id = args.run_id
    main_config_path = args.main_config.resolve()
    output = Path(yaml.safe_load(main_config_path.read_text(encoding="utf-8"))["output_root"]) / run_id
    status_path = output / "convex_pipeline_status.json"
    python = sys.executable
    main_cli = str(HERE / "experiment_1_main" / "run.py")
    selection_cli = str(HERE / "experiment_2_model_selection" / "run.py")
    scene_cli = str(HERE / "scene_evaluation" / "run.py")
    adaptation_cli = str(HERE / "experiment_3_regional_adaptation" / "run.py")
    temporal_cli = str(HERE / "experiment_4_temporal_window" / "run.py")
    main_config = yaml.safe_load(main_config_path.read_text(encoding="utf-8"))
    configured_regions = {str(item["id"]) for item in main_config["regions"]}
    prepare_command = [python, main_cli, "prepare", "--run-id", run_id, "--config", str(main_config_path)]
    if args.track_selection_csv:
        prepare_command.extend(["--track-selection-csv", str(args.track_selection_csv.resolve())])
    if args.scene_match_days is not None:
        prepare_command.extend(["--scene-match-days", str(args.scene_match_days)])
    if args.prepared_source_run_id or args.rebuild_region:
        if not (args.prepared_source_run_id and args.rebuild_region):
            raise SystemExit("--prepared-source-run-id and --rebuild-region must be supplied together")
        if not (output / "prepared" / "reuse_lineage.json").exists():
            seed_prepared_regions(
                REPO / "outputs" / "ten_region_bathymetry",
                args.prepared_source_run_id,
                run_id,
                args.rebuild_region,
                configured_regions,
            )
        prepare_command.extend(["--regions", args.rebuild_region])
    adaptation_run_command = [python, adaptation_cli, "run", "--main-run-id", run_id,
                              "--main-config", str(main_config_path)]
    if args.adaptation_targets:
        adaptation_run_command.extend(["--targets", args.adaptation_targets])
    stages = [
        ("main_prepare", prepare_command),
        ("main_split", [python, main_cli, "split", "--run-id", run_id, "--config", str(main_config_path)]),
        # Determine the usable track/scene time window before comparing
        # regressors.  This temporal stage consumes the freshly matched
        # Proposed table and fixed spatial split from the two stages above.
        ("temporal_window_qc", [python, temporal_cli, "qc", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("temporal_window_prepare", [python, temporal_cli, "prepare", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("temporal_window_run", [python, temporal_cli, "run", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("temporal_window_report", [python, temporal_cli, "report", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        # The fold-level ConvexBlend models and identical CUDEM anchor files
        # are prerequisites for both subsequent comparisons.
        ("main_cv_train", [python, main_cli, "train", "--run-id", run_id, "--config", str(main_config_path)]),
        ("main_cv_report", [python, main_cli, "report", "--run-id", run_id, "--config", str(main_config_path)]),
        # With the temporal protocol fixed, compare the eight downstream
        # regressors on exactly the same held-out spatial support.
        ("model_selection_qc", [python, selection_cli, "qc", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("model_selection_run", [python, selection_cli, "run", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("model_selection_report", [python, selection_cli, "report", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        # The main CV report already compares all seven classification/
        # refraction-correction inputs with one ConvexBlend per method and
        # identical CUDEM anchor pixels.  Do not launch a second, redundant
        # classification-method training experiment here.
        # Only after the temporal window and regressor comparison are
        # available do we run the primary full-data deployment model and
        # scene products. main_cv_train above is outer-fold training only.
        ("main_train_final", [python, main_cli, "train-final", "--run-id", run_id, "--config", str(main_config_path)]),
        ("scene_evaluation_qc", [python, scene_cli, "qc", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("scene_evaluation_run", [python, scene_cli, "run", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("scene_evaluation_report", [python, scene_cli, "report", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("adaptation_qc", [python, adaptation_cli, "qc", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
        ("adaptation_run", adaptation_run_command),
        ("adaptation_report", [python, adaptation_cli, "report", "--main-run-id", run_id, "--main-config", str(main_config_path)]),
    ]
    if args.start_at:
        names = [stage for stage, _ in stages]
        if args.start_at not in names:
            raise SystemExit(f"unknown stage {args.start_at}; choose from {names}")
        stages = stages[names.index(args.start_at):]
    for stage, command in stages:
        run_stage(status_path, run_id, stage, command)
    write_status(status_path, run_id, "pipeline", "completed", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
