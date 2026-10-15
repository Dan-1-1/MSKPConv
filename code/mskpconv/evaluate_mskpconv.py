import argparse
import glob
import os
from pathlib import Path

import torch

from config import Config
from mskpconv_model import PhysicsKPConvNet
from train_mskpconv import set_seed, test_per_file


def run_standalone_test(model_path, test_dir=None):
    """Load a trained model and run file-wise inference on the test directory."""
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    test_root = test_dir or Config.TEST_DATA_DIR
    test_files = sorted(glob.glob(os.path.join(test_root, "*.csv")))
    if not test_files:
        print("No test CSV files found.")
        return

    if not os.path.exists(model_path):
        print(f"Model weight not found: {model_path}")
        return

    model = PhysicsKPConvNet(num_classes=Config.NUM_CLASSES, dropout=Config.DROPOUT).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    try:
        test_per_file(model, test_files, device)
        print("Standalone test finished.")
    except Exception as e:
        print(f"Standalone test failed: {e}")


def main():
    release_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate the released MSKPConv checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=str(release_root / "models" / "classification" / "mskpconv" / "mskpconv_best.pth"),
    )
    parser.add_argument("--test-dir", default=Config.TEST_DATA_DIR)
    args = parser.parse_args()
    run_standalone_test(model_path=args.checkpoint, test_dir=args.test_dir)


if __name__ == "__main__":
    main()
