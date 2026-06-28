"""
Generate land mask raster from CUDEM aligned DEM.

For each region, reads the aligned CUDEM (NAVD88 elevation),
creates a binary land mask where:
    1 = land (elevation > 0)
    0 = water / bathymetry (elevation <= 0)
    255 = NoData

Saves land_mask.tif next to cudem_aligned.tif.

Usage:
    python generate_land_mask.py
"""
from pathlib import Path

import numpy as np
import rasterio


DATA_ROOT = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data")
REGIONS = ["Florida Bay", "Key Largo", "Key West", "Marathon"]


def safe_name(region: str) -> str:
    return region.replace(" ", "_")


def generate_land_mask(region: str) -> Path | None:
    """Create land_mask.tif from cudem_aligned.tif for a single region."""
    region_safe = safe_name(region)
    cudem_path = DATA_ROOT / region / "Cudem" / f"{region_safe}_cudem_aligned.tif"
    out_path = DATA_ROOT / region / "Cudem" / f"{region_safe}_land_mask.tif"

    if not cudem_path.exists():
        print(f"[SKIP] CUDEM not found: {cudem_path}")
        return None

    with rasterio.open(cudem_path) as src:
        cudem = src.read(1).astype("float32")

        # CUDEM convention: land > 0, water < 0, nodata = -9999 or similar
        land_mask = np.full(cudem.shape, 255, dtype="uint8")
        land_mask[(cudem > 0) & np.isfinite(cudem)] = 1   # land
        land_mask[(cudem <= 0) & np.isfinite(cudem)] = 0  # water

        profile = src.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(land_mask, 1)

    land_count = int((land_mask == 1).sum())
    water_count = int((land_mask == 0).sum())
    print(f"[OK] {region}: land={land_count:,}, water={water_count:,} -> {out_path}")
    return out_path


def main():
    print("=" * 70)
    print("Land Mask Generation from CUDEM")
    print("=" * 70)
    for region in REGIONS:
        generate_land_mask(region)
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
