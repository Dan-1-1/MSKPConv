import glob
import os

import torch

from config import Config
from mskpconv_model import PhysicsKPConvNet
from train_mskpconv import set_seed, test_per_file


def run_standalone_test(model_path="best_model.pth"):
    """Load a trained model and run file-wise inference on the test directory."""
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    test_files = glob.glob(os.path.join(Config.TEST_DATA_DIR, "*.csv"))
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


if __name__ == "__main__":
    WEIGHT_FILE = os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth")
    run_standalone_test(model_path=WEIGHT_FILE)
