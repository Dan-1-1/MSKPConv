"""Generate the public, path-independent split manifests from experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROW_COLUMNS = [
    "track_group_id",
    "scene_id",
    "img_row",
    "img_col",
    "region_id",
    "method",
    "Latitude",
    "Longitude",
    "fold",
    "split_fold_0",
    "split_fold_1",
    "split_fold_2",
    "split_fold_3",
    "split_fold_4",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def classification_manifest(data_root: Path, output: Path) -> None:
    rows = []
    for split in ("train", "validation", "test"):
        for path in sorted((data_root / split).glob("*.csv")):
            rows.append(
                {
                    "split": split,
                    "file_name": path.name,
                    "relative_path": f"{split}/{path.name}",
                    "rows": count_csv_rows(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No classification CSVs found below {data_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, lineterminator="\n")


def inversion_manifests(split_root: Path, output_root: Path) -> None:
    frames = []
    for path in sorted(split_root.glob("*/*.parquet")):
        frame = pd.read_parquet(path, columns=ROW_COLUMNS).reset_index(names="source_row_index")
        frame.insert(0, "source_file", path.relative_to(split_root).as_posix())
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No inversion split Parquets found below {split_root}")
    combined = pd.concat(frames, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_root / "row_splits.parquet", index=False, compression="zstd")
    track_folds = (
        combined[["region_id", "method", "track_group_id", "fold"]]
        .drop_duplicates()
        .sort_values(["region_id", "method", "fold", "track_group_id"])
    )
    track_folds.to_csv(output_root / "track_folds.csv", index=False, lineterminator="\n")
    summary = {
        "source_parquet_files": len(frames),
        "row_assignments": len(combined),
        "regions": sorted(combined["region_id"].unique().tolist()),
        "methods": sorted(combined["method"].unique().tolist()),
        "outer_folds": 5,
        "columns": combined.columns.tolist(),
        "row_splits_sha256": sha256(output_root / "row_splits.parquet"),
        "track_folds_sha256": sha256(output_root / "track_folds.csv"),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--inversion-split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("splits"))
    args = parser.parse_args()
    classification_manifest(
        args.classification_root,
        args.output_root / "photon_classification" / "files.csv",
    )
    inversion_manifests(args.inversion_split_root, args.output_root / "convexblend")


if __name__ == "__main__":
    main()
