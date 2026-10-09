# ATL03 dataset metadata

`ATL03_Track_Metadata.csv` contains one row for each of the 220 labeled ATL03 track CSV files. It records split, ATL03 release/version, UTC acquisition date/time, reference ground track (RGT), orbital cycle, photon count, and each track's latitude/longitude extent. The filename-derived acquisition dates span 2018-10-14 to 2025-04-26; the study documentation describes the planned acquisition window as 2018–2026. The current collection contains Releases 005, 006, and 007, rather than Release 007 exclusively.

`ATL03_Split_Summary.csv` reports the 220-track split totals. Photon counts are counted from labeled CSV rows. The 17 former `subsetted` entries were recovered from the server's Train/Val/Test folders and added to the repository; their metadata now contains measured photon counts and coordinate extents. Their server source files are Release 005, even though the historical index aliases contained `_007_`; the metadata records the recovered server release to avoid misreporting the product version.

| Field | Definition |
|---|---|
| `FileName` | Labeled photon CSV filename |
| `Split` | `Train`, `Val`, or `Test` |
| `ATL03_Release` / `ATL03_Version` | ATL03 product release and file version parsed from the granule ID |
| `AcquisitionDate` / `GranuleDateTime` | UTC acquisition date/time parsed from the granule ID |
| `RGT` / `ReferenceGroundTrack` | Reference ground track number |
| `Cycle` | ICESat-2 repeat cycle |
| `PhotonCount` | Number of data rows in the labeled track CSV |
| `LatitudeMin` / `LatitudeMax` | Track latitude range in degrees |
| `LongitudeMin` / `LongitudeMax` | Track longitude range in degrees |
| `GeographicExtent` | Human-readable coordinate extent |

## Ten regional DEM sites

`ATL03_DEM_Region_Track_Metadata.csv` lists the ATL03 files found in the ten requested DEM regions: six numbered `Six_Regions` folders plus Florida Bay, Key Largo, Key West, and Marathon. It records every selected ATL03 filename together with release/version, RGT, cycle, acquisition date/time, source path, and coordinate tokens parsed from the filename.

| Region | ATL03 CSV files | ATL03 release/version | RGT / Cycle | Acquisition date/time (UTC) |
|---|---:|---|---|---|
| 01_Anegada | 43 | See region metadata | See region metadata | See region metadata |
| 02_Manua | 17 | See region metadata | See region metadata | See region metadata |
| 03_Cocos_Babe_Guam | 101 | See region metadata | See region metadata | See region metadata |
| 04_Kahului | 51 | See region metadata | See region metadata | See region metadata |
| 05_Saipan_West_Coast | 97 | See region metadata | See region metadata | See region metadata |
| 06_Kaneohe_Bay | 79 | See region metadata | See region metadata | See region metadata |
| Florida Bay | 22 | See region metadata | See region metadata | See region metadata |
| Key Largo | 150 | See region metadata | See region metadata | See region metadata |
| Key West | 65 | See region metadata | See region metadata | See region metadata |
| Marathon | 60 | See region metadata | See region metadata | See region metadata |
