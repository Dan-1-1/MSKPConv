"""Read-only audit of the NOAA DEM source selected for the local Tutuila raster.

The existing Tutuila raster was exported from NOAA's DEM_all ImageServer without
source metadata.  This script requests the candidate high-resolution source
layers at the exact same extent/grid and compares the returned values with the
local raster.  It does not overwrite any data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile


ROOT = Path(r"H:\ATL24\PhisicKPConvNet")
LOCAL = ROOT / r"validation_area/data/Six_Regions/07_Tutuila_Pala_Lagoon/Cudem/07_Tutuila_Pala_Lagoon_cudem.tif"
URL = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage"
CANDIDATES = ("tutuila_as_1_3", "pago_pago_1-3s", "pago_pago_3s")


def request_layer(name: str, bounds, width: int, height: int) -> np.ndarray:
    left, bottom, right, top = bounds
    params = {
        "bbox": f"{left},{bottom},{right},{top}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "mosaicRule": json.dumps({"where": f"Name = '{name}'"}),
        "f": "image",
    }
    response = requests.get(URL, params=params, timeout=180)
    response.raise_for_status()
    if "image" not in response.headers.get("Content-Type", "").lower():
        raise RuntimeError(response.text[:500])
    with MemoryFile(response.content) as mem:
        with mem.open() as src:
            return src.read(1).astype("float64")


def main() -> None:
    with rasterio.open(LOCAL) as src:
        local = src.read(1).astype("float64")
        bounds, width, height = src.bounds, src.width, src.height
    print(f"local={LOCAL}")
    print(f"grid={width}x{height} bounds={bounds}")
    for name in CANDIDATES:
        try:
            candidate = request_layer(name, bounds, width, height)
            valid = np.isfinite(local) & np.isfinite(candidate) & (local > -9000) & (candidate > -9000)
            if valid.sum() < 10:
                print(f"{name}: insufficient valid pixels ({valid.sum()})")
                continue
            corr = float(np.corrcoef(local[valid], candidate[valid])[0, 1])
            mae = float(np.mean(np.abs(local[valid] - candidate[valid])))
            delta = float(np.mean(local[valid] - candidate[valid]))
            print(f"{name}: n={valid.sum()} corr={corr:.8f} mae={mae:.8f} mean_local_minus_source={delta:.8f}")
        except Exception as exc:
            print(f"{name}: ERROR {exc}")


if __name__ == "__main__":
    main()
