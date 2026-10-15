# Representative-scene CUDEM evaluation

This experiment complements, rather than replaces, the track-blocked main
experiment. It applies the final pooled ConvexBlend model trained from the
Proposed photon classification/refraction source to the ten preselected
Sentinel-2 representative scenes and compares it against external CUDEM
bathymetry.

Within each region, Proposed–ConvexBlend is scored only where its prediction is
finite and reprojected CUDEM represents an underwater cell. CUDEM is treated
as elevation and converted with `depth = -elevation`; the sign is not inferred
from the scene median because land dominates several representative scenes.
No upper-depth limit is applied: each representative scene uses its complete
CUDEM bathymetric range, and out-of-training-range model predictions are retained
so extrapolation errors remain visible in the metrics.

```powershell
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/scene_evaluation/run.py qc --main-run-id 20261007T003000Z_cudem_anchor
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/scene_evaluation/run.py run --main-run-id 20261007T003000Z_cudem_anchor
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/scene_evaluation/run.py report --main-run-id 20261007T003000Z_cudem_anchor
```

Metrics and figures are written under:

```text
outputs/ten_region_bathymetry/<run-id>/scene_evaluation/representative_scene_cudem_v1/
```

Large depth, uncertainty, CUDEM-reference, and common-mask GeoTIFFs are written
under the configured `raster_output_root` on G:.
