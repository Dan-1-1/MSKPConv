import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import Config
from dataset import PhotonFileTestDataset
from mskpconv_model import PhysicsKPConvNet
from visualize_photon_predictions import plot_single_file_result, save_predictions_to_csv


def region_prediction_dir(region_dir):
    return os.path.join(region_dir, "Predictions")


@torch.no_grad()
def run_region_prediction(model, region_dir, device, batch_size=None):
    test_files = glob.glob(os.path.join(region_dir, "*.csv"))
    if not test_files:
        print(f"[Skip] No CSV files found in {region_dir}")
        return

    pred_dir = region_prediction_dir(region_dir)
    os.makedirs(pred_dir, exist_ok=True)

    dataset = PhotonFileTestDataset(test_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    file_votes = {fi: np.zeros((data["num_points"], Config.NUM_CLASSES)) for fi, data in enumerate(dataset.file_data)}
    file_metrics = []

    model.eval()
    for pos, features_local, _, _, _, file_indices, point_indices in loader:
        pos = pos.to(device)
        features_local = features_local.to(device)
        logits = model(pos, features_local)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        file_indices = file_indices.numpy()
        point_indices = point_indices.numpy()

        for b in range(probs.shape[0]):
            fi = int(file_indices[b])
            for n in range(probs.shape[2]):
                pi = int(point_indices[b, n])
                file_votes[fi][pi] += probs[b, :, n]

    for fi, data in enumerate(dataset.file_data):
        final_pred_labels = np.argmax(file_votes[fi], axis=1)
        labels = data["labels"]
        acc = (final_pred_labels == labels).mean()
        macro_f1 = f1_score(
            labels,
            final_pred_labels,
            labels=list(range(Config.NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
        class_f1s = f1_score(
            labels,
            final_pred_labels,
            labels=list(range(Config.NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        plot_single_file_result(
            data["x_raw"],
            data["z_raw"],
            labels,
            final_pred_labels,
            data["filename"],
            save_dir=pred_dir,
            macro_f1=macro_f1,
        )
        row = {
            "filename": data["filename"],
            "num_points": int(data["num_points"]),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
        }
        for cls_idx, cls_name in enumerate(Config.CLASS_NAMES):
            row[f"f1_{cls_name}"] = float(class_f1s[cls_idx]) if cls_idx < len(class_f1s) else 0.0
        file_metrics.append(row)

    save_predictions_to_csv(
        file_data_list=dataset.file_data,
        file_votes=file_votes,
        test_dir=region_dir,
        save_dir=pred_dir,
    )

    metrics_path = os.path.join(pred_dir, "file_metrics.csv")
    pd.DataFrame(file_metrics).to_csv(metrics_path, index=False)
    print(f"[Done] Region: {region_dir}")
    print(f"       Predictions: {pred_dir}")
    print(f"       Metrics: {metrics_path}")


def run_all_regions(model_path=None, region_dirs=None, batch_size=None, device=None):
    model_path = model_path or os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth")
    region_dirs = region_dirs or Config.PREDICT_REGION_DIRS
    if device is None:
        device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    model = PhysicsKPConvNet(num_classes=Config.NUM_CLASSES, dropout=Config.DROPOUT).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    for region_dir in region_dirs:
        run_region_prediction(model, region_dir, device, batch_size=batch_size or Config.BATCH_SIZE)


def main():
    run_all_regions()


if __name__ == "__main__":
    main()
