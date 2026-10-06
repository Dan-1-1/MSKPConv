import glob
import json
import os

import h5py
import numpy as np
from tqdm import tqdm

from config import Config
from preprocess_atl03_features import load_and_preprocess_single_csv


def save_to_hdf5(raw_dir, h5_path):
    """Preprocess CSV files under a directory and store them in one HDF5 file."""
    if not os.path.exists(raw_dir):
        print(f"Skip: missing directory {raw_dir}")
        return

    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print(f"Skip: no CSV files in {raw_dir}")
        return

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    diagnostics_rows = []
    with h5py.File(h5_path, "w") as h5_file:
        for fp in tqdm(csv_files, desc=os.path.basename(h5_path)):
            fname = os.path.basename(fp)
            try:
                data = load_and_preprocess_single_csv(fp)
                group = h5_file.create_group(fname)
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value, compression="gzip", compression_opts=4)
                    else:
                        group.attrs[key] = value
                diagnostics = json.loads(data.get("feature_diagnostics_json", "{}"))
                for stat in diagnostics.get("query_stats", []):
                    diagnostics_rows.append(
                        {
                            "file": fname,
                            "max_pull_factor": diagnostics.get("max_pull_factor"),
                            **stat,
                        }
                    )
            except Exception as e:
                print(f"Failed: {fname} -> {e}")

    if diagnostics_rows:
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        diagnostics_path = os.path.join(
            Config.SAVE_DIR,
            f"feature_diagnostics_{os.path.basename(h5_path)}.csv",
        )
        import pandas as pd

        pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, index=False)
        print(f"Feature diagnostics saved to: {diagnostics_path}")


if __name__ == "__main__":
    save_to_hdf5(Config.DATA_DIR, Config.HDF5_TRAIN_PATH)
    save_to_hdf5(Config.VAL_DATA_DIR, Config.HDF5_VAL_PATH)
    save_to_hdf5(Config.TEST_DATA_DIR, Config.HDF5_TEST_PATH)
    print("HDF5 cache generation finished.")
