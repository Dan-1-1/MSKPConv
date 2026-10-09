# ATL03 dataset metadata

The renamed metadata index is `ATL03_Track_Metadata.csv`. It contains one row for each of the 220 labeled ATL03 track CSV files, including split, ATL03 release/version, acquisition date, reference ground track (RGT), and orbital cycle. The filename-derived acquisition dates span 2018-10-14 to 2025-04-26; the study documentation describes the planned acquisition window as 2018–2026. The index records the source values and therefore shows that the current collection contains Releases 005, 006, and 007 (not Release 007 exclusively).

| Field | Definition |
|---|---|
| `FileName` | Labeled photon CSV filename |
| `Split` | `Train`, `Val`, or `Test` |
| `ATL03_Release` / `ATL03_Version` | ATL03 product release and file version parsed from the granule ID |
| `AcquisitionDate` / `GranuleDateTime` | UTC acquisition date/time parsed from the granule ID |
| `RGT` / `ReferenceGroundTrack` | Reference ground track number |
| `Cycle` | ICESat-2 repeat cycle |

## Ten regional DEM sites

The following table is the recommended companion table for the manuscript. The `Region` and `CSV files` cells must be filled from the study's authoritative ten-site map/region list; the metadata values should be copied from `ATL03_Track_Metadata.csv` rather than inferred from a site name. This prevents a track from being assigned to the wrong regional DEM site when a filename does not contain a place name.

| Region | CSV files used | ATL03 release/version | RGT | Cycle | Acquisition date(s) (UTC) |
|---|---|---|---|---|---|
| Region 1 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 2 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 3 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 4 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 5 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 6 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 7 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 8 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 9 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
| Region 10 | To be mapped from the ten-site list | See metadata index | See metadata index | See metadata index | See metadata index |
