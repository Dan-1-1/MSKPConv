"""Verify released weights, split artifacts, and ConvexBlend invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    split_manifest_path = ROOT / "splits" / "convexblend" / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    row_path = ROOT / "splits" / "convexblend" / "row_splits.parquet"
    track_path = ROOT / "splits" / "convexblend" / "track_folds.csv"
    assert sha256(row_path) == split_manifest["row_splits_sha256"]
    assert sha256(track_path) == split_manifest["track_folds_sha256"]
    rows = pd.read_parquet(row_path)
    assert len(rows) == split_manifest["row_assignments"]
    assert set(rows["fold"].unique()) <= set(range(5))
    allowed = {"train", "validation", "test", "excluded_spatial_buffer"}
    for fold in range(5):
        assert set(rows[f"split_fold_{fold}"].dropna().unique()) <= allowed

    weights_manifest = json.loads(
        (ROOT / "models" / "WEIGHTS_MANIFEST.json").read_text(encoding="utf-8")
    )
    for item in weights_manifest["repository_files"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert sha256(path) == item["sha256"], path

    blend_path = ROOT / "models" / "convexblend" / "final" / "Proposed" / "ConvexBlend" / "final.json"
    blend = json.loads(blend_path.read_text(encoding="utf-8"))
    weights = np.asarray(blend["weights"], dtype="float64")
    assert len(blend["base_regressors"]) == 5
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)
    print(f"Verified {len(rows):,} inversion row assignments and all repository weights.")


if __name__ == "__main__":
    main()
