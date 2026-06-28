import glob
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from config import Config
from dataset import PhotonFileTestDataset, get_dataloaders
from losses import DynamicWeightedCELoss, PhysicsLovaszLoss
from mskpconv_model import PhysicsKPConvNet
from visualize_photon_predictions import plot_single_file_result, plot_training_curves, save_predictions_to_csv


EPOCH_LOG_FMT_STR = (
    "{epoch:>5s} | {tr_loss:>8s} | {focal:>8s} | {lovasz:>8s} | "
    "{tr_acc:>7s} | {val_loss:>8s} | {f1_mean:>7s} | {val_acc:>7s} | "
    "{noise:>7s} | {surface:>7s} | {land:>7s} | {seabed:>7s} | {lr:>10s} | {time:>8s}"
)
LOG_SEPARATOR = "-" * len(
    EPOCH_LOG_FMT_STR.format(
        epoch="Epoch",
        tr_loss="Tr_Loss",
        focal="Focal",
        lovasz="Lovasz",
        tr_acc="Tr_Acc",
        val_loss="Val_Loss",
        f1_mean="F1_Mean",
        val_acc="Val_Acc",
        noise="Noise",
        surface="Surfac",
        land="Land",
        seabed="Seabed",
        lr="LR",
        time="Time",
    )
)


def set_seed(seed=Config.SEED):
    """Set deterministic random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(all_preds, all_labels, class_names=Config.CLASS_NAMES):
    """Compute classification report and confusion matrix on flattened labels."""
    labels = list(range(Config.NUM_CLASSES))
    report = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    return report, cm


def derive_metrics_from_cm(cm):
    """Derive OA, mIoU and per-class precision/recall/F1/IoU from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    oa = float(np.trace(cm) / total) if total > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []
    ious = []
    accuracies = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        iou = tp / max(tp + fp + fn, 1.0)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        ious.append(float(iou))
        accuracies.append(float(accuracy))

    return {
        "OA": oa,
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "iou": ious,
        "accuracy": accuracies,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return average loss/accuracy components."""
    model.train()
    total_loss = 0.0
    total_focal_loss = 0.0
    total_lovasz_loss = 0.0
    total_correct = 0
    total_points = 0

    for pos, features_local, labels in loader:
        pos = pos.to(device)
        depth = pos[:, :, 1]
        features_local = features_local.to(device)
        labels = labels.to(device)

        logits = model(pos, features_local)
        base_loss, focal_depth_loss, lovasz_loss = criterion(logits, labels, depth)
        loss = base_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_focal_loss += focal_depth_loss.item()
        total_lovasz_loss += lovasz_loss.item()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_points += labels.numel()

    n = max(len(loader), 1)
    return (
        total_loss / n,
        total_correct / max(total_points, 1),
        total_focal_loss / n,
        total_lovasz_loss / n,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model and return loss, accuracy, macro-F1, class-F1, predictions and labels."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for pos, features_local, labels in loader:
        pos = pos.to(device)
        depth = pos[:, :, 1]
        features_local = features_local.to(device)
        labels = labels.to(device)

        logits = model(pos, features_local)
        base_loss, *_ = criterion(logits, labels, depth)
        loss = base_loss
        total_loss += loss.item()

        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy().reshape(-1))
        all_labels.append(labels.cpu().numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / max(len(loader), 1)
    acc = (all_preds == all_labels).mean()
    f1_mean = f1_score(
        all_labels,
        all_preds,
        labels=list(range(Config.NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    class_f1s = f1_score(
        all_labels,
        all_preds,
        labels=list(range(Config.NUM_CLASSES)),
        average=None,
        zero_division=0,
    )

    return avg_loss, acc, f1_mean, class_f1s, all_preds, all_labels


@torch.no_grad()
def test_per_file(model, test_files, device):
    """Run file-wise inference with overlap voting and export plots/CSVs."""
    model.eval()
    file_ds = PhotonFileTestDataset(test_files)
    file_loader = DataLoader(file_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    file_votes = {
        fi: np.zeros((data["num_points"], Config.NUM_CLASSES))
        for fi, data in enumerate(file_ds.file_data)
    }
    file_metrics = []

    for pos, features_local, _, _, _, file_indices, point_indices in file_loader:
        pos = pos.to(device)
        features_local = features_local.to(device)
        logits = model(pos, features_local)

        preds_probs = torch.softmax(logits, dim=1).cpu().numpy()
        file_indices = file_indices.numpy()
        point_indices = point_indices.numpy()

        for b in range(preds_probs.shape[0]):
            fi = file_indices[b]
            for n in range(preds_probs.shape[2]):
                pi = point_indices[b, n]
                file_votes[fi][pi] += preds_probs[b, :, n]

    for fi, data in enumerate(file_ds.file_data):
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
            macro_f1=macro_f1,
        )
        row = {
            "filename": data["filename"],
            "num_points": int(data["num_points"]),
            "accuracy": acc,
            "macro_f1": macro_f1,
        }
        for cls_idx, cls_name in enumerate(Config.CLASS_NAMES):
            row[f"f1_{cls_name}"] = class_f1s[cls_idx] if cls_idx < len(class_f1s) else 0.0
        file_metrics.append(row)

    save_predictions_to_csv(
        file_data_list=file_ds.file_data,
        file_votes=file_votes,
        test_dir=Config.TEST_DATA_DIR,
        save_dir=Config.PREDICTION_SAVE_DIR,
    )
    os.makedirs(Config.PREDICTION_SAVE_DIR, exist_ok=True)
    metrics_path = os.path.join(Config.PREDICTION_SAVE_DIR, "file_metrics.csv")
    pd.DataFrame(file_metrics).to_csv(metrics_path, index=False)
    print(pd.DataFrame(file_metrics).to_string(index=False))
    print(f"Per-file metrics saved to: {metrics_path}")
    print("File-wise inference finished.")


def setup_logger(save_dir):
    """Create a logger that writes both to a timestamped file and to console."""
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = os.path.join(save_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def save_config(config_obj, save_dir):
    """Persist Config class attributes to JSON for experiment reproducibility."""
    config_dict = {k: v for k, v in config_obj.__dict__.items() if not k.startswith("__")}
    config_path = os.path.join(save_dir, "config_params.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    return config_path


def count_trainable_parameters(model):
    """Return the number of trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_epoch_header(logger):
    """Print the tabular epoch log header."""
    logger.info(LOG_SEPARATOR)
    logger.info(
        EPOCH_LOG_FMT_STR.format(
            epoch="Epoch",
            tr_loss="Tr_Loss",
            focal="Focal",
            lovasz="Lovasz",
            tr_acc="Tr_Acc",
            val_loss="Val_Loss",
            f1_mean="F1_Mean",
            val_acc="Val_Acc",
            noise="Noise",
            surface="Surfac",
            land="Land",
            seabed="Seabed",
            lr="LR",
            time="Time",
        )
    )
    logger.info(LOG_SEPARATOR)


def log_epoch_row(
    logger,
    epoch,
    train_loss,
    focal_loss,
    lovasz_loss,
    train_acc,
    val_loss,
    f1_mean,
    val_acc,
    f1_noise,
    f1_surface,
    f1_land,
    f1_seabed,
    lr,
    elapsed,
):
    """Print one formatted epoch metric row."""
    logger.info(
        EPOCH_LOG_FMT_STR.format(
            epoch=f"{epoch:d}",
            tr_loss=f"{train_loss:.4f}",
            focal=f"{focal_loss:.4f}",
            lovasz=f"{lovasz_loss:.4f}",
            tr_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            f1_mean=f"{f1_mean:.4f}",
            val_acc=f"{val_acc:.4f}",
            noise=f"{f1_noise:.4f}",
            surface=f"{f1_surface:.4f}",
            land=f"{f1_land:.4f}",
            seabed=f"{f1_seabed:.4f}",
            lr=f"{lr:.6f}",
            time=f"{elapsed:.1f}s",
        )
    )


def main():
    """Run full training pipeline including validation, checkpointing, and final testing."""
    set_seed()
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    logger = setup_logger(Config.SAVE_DIR)
    logger.info("============================== 实验启动 ==============================")
    logger.info("设备信息: %s", device)

    config_path = save_config(Config, Config.SAVE_DIR)
    logger.info("超参数配置已保存至 %s", config_path)

    train_loader, val_loader, test_loader = get_dataloaders()
    logger.info(
        "数据加载完成。训练集样本量: %d, 验证集样本量: %d, 测试集样本量: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = PhysicsKPConvNet(num_classes=Config.NUM_CLASSES, dropout=Config.DROPOUT).to(device)
    logger.info("模型参数量: %s", f"{count_trainable_parameters(model):,}")

    if Config.PHYSICAL_LOSS:
        criterion = PhysicsLovaszLoss().to(device)
        logger.info("使用 Phisic Loss 和 Lovasz Loss 作为物理损失分量")
    else:
        criterion = DynamicWeightedCELoss().to(device)
        logger.info("使用动态加权 CE Loss 和 Dice Loss")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=Config.WARMUP_EPOCHS,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.SCHEDULER_T0,
        T_mult=1,
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[Config.WARMUP_EPOCHS],
    )

    best_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth")

    history = {
        "train_loss": [],
        "focal_loss": [],
        "lovasz_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_oa": [],
        "val_miou": [],
        "f1": [],
        "val_acc": [],
        "f1_noise": [],
        "f1_surface": [],
        "f1_land": [],
        "f1_seabed": [],
        "seabed_acc": [],
        "seabed_precision": [],
        "seabed_recall": [],
        "seabed_iou": [],
        "lr": [],
    }

    log_epoch_header(logger)

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        if hasattr(criterion, "step_lovasz_weight"):
            criterion.step_lovasz_weight(epoch, Config.EPOCHS)

        train_loss, train_acc, focal_loss, lovasz_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc, f1_mean, class_f1s, val_pred, val_true = evaluate(
            model, val_loader, criterion, device
        )
        _, val_cm = compute_metrics(val_pred, val_true)
        val_stats = derive_metrics_from_cm(val_cm)

        f1_n = class_f1s[0] if len(class_f1s) > 0 else 0.0
        f1_s = class_f1s[1] if len(class_f1s) > 1 else 0.0
        f1_l = class_f1s[2] if len(class_f1s) > 2 else 0.0
        f1_b = class_f1s[3] if len(class_f1s) > 3 else 0.0
        val_oa = val_stats["OA"]
        val_miou = val_stats["mIoU"]
        seabed_acc = val_stats["accuracy"][3] if len(val_stats["accuracy"]) > 3 else 0.0
        seabed_precision = val_stats["precision"][3] if len(val_stats["precision"]) > 3 else 0.0
        seabed_recall = val_stats["recall"][3] if len(val_stats["recall"]) > 3 else 0.0
        seabed_iou = val_stats["iou"][3] if len(val_stats["iou"]) > 3 else 0.0

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(
            history.keys(),
            [
                train_loss,
                focal_loss,
                lovasz_loss,
                train_acc,
                val_loss,
                val_oa,
                val_miou,
                f1_mean,
                val_acc,
                f1_n,
                f1_s,
                f1_l,
                f1_b,
                seabed_acc,
                seabed_precision,
                seabed_recall,
                seabed_iou,
                lr,
            ],
        ):
            history[k].append(v)

        pd.DataFrame(history).to_csv(os.path.join(Config.SAVE_DIR, "train_history.csv"), index=False)

        elapsed = time.time() - epoch_start
        log_epoch_row(
            logger,
            epoch,
            train_loss,
            focal_loss,
            lovasz_loss,
            train_acc,
            val_loss,
            f1_mean,
            val_acc,
            f1_n,
            f1_s,
            f1_l,
            f1_b,
            lr,
            elapsed,
        )

        if f1_mean > best_f1:
            best_f1 = f1_mean
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info("   >>> 更新最佳模型: Epoch %d, 整体 F1=%.4f (海底 F1=%.4f)", epoch, f1_mean, f1_b)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                logger.warning("早停触发：连续 %d 轮未提升。", Config.EARLY_STOP_PATIENCE)
                break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc, test_f1, _, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    logger.info("Test | loss %.4f | f1 %.4f | acc %.4f", test_loss, test_f1, test_acc)

    report, cm = compute_metrics(test_preds, test_labels)
    logger.info("\n%s", report)
    logger.info("\n%s", cm)

    plot_training_curves(history, save_dir=Config.SAVE_DIR_PLOT)
    test_files = glob.glob(os.path.join(Config.TEST_DATA_DIR, "*.csv"))
    test_per_file(model, test_files, device)

    if getattr(Config, "AUTO_PREDICT_REGIONS", False):
        from auto_predict_regions import run_all_regions

        logger.info("开始自动执行四个区域预测输出...")
        run_all_regions(
            model_path=save_path,
            region_dirs=Config.PREDICT_REGION_DIRS,
            batch_size=Config.BATCH_SIZE,
            device=device,
        )
        logger.info("四个区域自动预测输出完成。")


if __name__ == "__main__":
    main()
