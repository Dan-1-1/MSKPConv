import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from validate_all_regions_cudem_correlation import (
    REGIONS,
    collect_matches,
    compute_metrics,
    copy_selected_files,
    iter_prediction_csvs,
    safe_output_name,
)


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("No image")

    def set_image(self, image_path: Path | None):
        if image_path is None or not image_path.exists():
            self._pixmap = None
            self.setText("Correlation image not found")
            self.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._pixmap = None
            self.setText(f"Cannot load image:\n{image_path}")
            self.setPixmap(QPixmap())
            return
        self._pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class ManualSelectionWindow(QMainWindow):
    def __init__(self, initial_region: str | None, chunksize: int):
        super().__init__()
        self.chunksize = chunksize
        self.csv_paths: list[Path] = []
        self.metrics_cache: dict[Path, dict] = {}
        self.current_region = None
        self.current_index = -1

        self.setWindowTitle("Manual ICESat-2 Prediction Selection")
        self.resize(1120, 760)

        self.region_box = QComboBox()
        self.region_box.addItems(sorted(REGIONS))
        if initial_region in REGIONS:
            self.region_box.setCurrentText(initial_region)
        self.region_box.currentTextChanged.connect(self.load_region)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.show_index)

        self.image_label = ImageLabel()
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setMinimumWidth(300)
        self.info_label.setWordWrap(True)

        self.select_button = QPushButton("选择并复制")
        self.skip_button = QPushButton("跳过/下一张")
        self.prev_button = QPushButton("上一张")
        self.next_button = QPushButton("下一张")
        self.refresh_button = QPushButton("刷新")
        self.open_selected_button = QPushButton("打开 selected 文件夹")
        self.clear_selected_button = QPushButton("清空 Predictions_selected")
        self.delete_group_button = QPushButton("删除当前已复制文件组")

        self.select_button.clicked.connect(self.select_current)
        self.skip_button.clicked.connect(self.next_file)
        self.prev_button.clicked.connect(self.prev_file)
        self.next_button.clicked.connect(self.next_file)
        self.refresh_button.clicked.connect(self.reload_current_region)
        self.open_selected_button.clicked.connect(self.open_selected_dir)
        self.clear_selected_button.clicked.connect(self.clear_selected_dir)
        self.delete_group_button.clicked.connect(self.delete_current_group)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("区域"))
        left_layout.addWidget(self.region_box)
        left_layout.addWidget(QLabel("预测文件"))
        left_layout.addWidget(self.file_list)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("指标"))
        right_layout.addWidget(self.info_label)
        right_layout.addStretch(1)
        right_layout.addWidget(self.select_button)
        right_layout.addWidget(self.skip_button)
        right_layout.addWidget(self.prev_button)
        right_layout.addWidget(self.next_button)
        right_layout.addWidget(self.refresh_button)
        right_layout.addWidget(self.open_selected_button)
        right_layout.addWidget(self.clear_selected_button)
        right_layout.addWidget(self.delete_group_button)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.image_label)
        splitter.addWidget(right)
        splitter.setSizes([280, 560, 280])

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

        self._add_shortcuts()
        self.load_region(self.region_box.currentText())

    def _add_shortcuts(self):
        toolbar = QToolBar("Shortcuts")
        toolbar.setVisible(False)
        self.addToolBar(toolbar)

        shortcuts = [
            ("Select", QKeySequence(Qt.Key_A), self.select_current),
            ("SelectEnter", QKeySequence(Qt.Key_Return), self.select_current),
            ("Next", QKeySequence(Qt.Key_D), self.next_file),
            ("NextRight", QKeySequence(Qt.Key_Right), self.next_file),
            ("PrevLeft", QKeySequence(Qt.Key_Left), self.prev_file),
            ("DeleteGroup", QKeySequence(Qt.Key_Delete), self.delete_current_group),
        ]
        for name, sequence, slot in shortcuts:
            action = QAction(name, self)
            action.setShortcut(sequence)
            action.triggered.connect(slot)
            toolbar.addAction(action)

    def region_paths(self):
        cfg = REGIONS[self.current_region]
        root = cfg["root"]
        return {
            "root": root,
            "pred_dir": root / "Predictions",
            "corr_dir": root / "ICESat2_CUDEM_Correlation",
            "selected_dir": root / "Predictions_selected",
            "cudem": cfg["cudem"],
        }

    def load_region(self, region_name: str):
        self.current_region = region_name
        self.metrics_cache.clear()
        paths = self.region_paths()
        self.file_list.clear()
        self.csv_paths = []
        self.current_index = -1

        if not paths["pred_dir"].exists():
            self.set_status(f"Predictions directory not found: {paths['pred_dir']}")
            self.image_label.set_image(None)
            self.info_label.setText("Predictions directory not found.")
            return

        self.csv_paths = iter_prediction_csvs(paths["pred_dir"])
        for csv_path in self.csv_paths:
            item = QListWidgetItem(self.list_label(csv_path))
            item.setData(Qt.UserRole, str(csv_path))
            self.file_list.addItem(item)

        if self.csv_paths:
            self.file_list.setCurrentRow(0)
        else:
            self.image_label.set_image(None)
            self.info_label.setText("No ATL03 CSV files found.")
        self.set_status(f"{region_name}: {len(self.csv_paths)} files")

    def reload_current_region(self):
        current_path = self.current_csv()
        self.load_region(self.current_region)
        if current_path in self.csv_paths:
            self.file_list.setCurrentRow(self.csv_paths.index(current_path))

    def list_label(self, csv_path: Path) -> str:
        paths = self.region_paths()
        corr_png = paths["corr_dir"] / safe_output_name(csv_path)
        selected_csv = paths["selected_dir"] / csv_path.name
        if selected_csv.exists():
            status = "已选择"
        elif not corr_png.exists():
            status = "缺少相关性图"
        else:
            status = "未处理"
        return f"[{status}] {csv_path.name}"

    def current_csv(self) -> Path | None:
        if 0 <= self.current_index < len(self.csv_paths):
            return self.csv_paths[self.current_index]
        return None

    def show_index(self, row: int):
        self.current_index = row
        csv_path = self.current_csv()
        if csv_path is None:
            return

        paths = self.region_paths()
        corr_png = paths["corr_dir"] / safe_output_name(csv_path)
        self.image_label.set_image(corr_png)
        metrics_text = self.metrics_text(csv_path)
        result_png = csv_path.with_name(f"{csv_path.stem}_result.png")
        selected_csv = paths["selected_dir"] / csv_path.name
        self.info_label.setText(
            f"区域: {self.current_region}\n"
            f"序号: {row + 1} / {len(self.csv_paths)}\n\n"
            f"CSV:\n{csv_path.name}\n\n"
            f"相关性图:\n{corr_png.name}\n"
            f"结果图: {result_png.name}\n\n"
            f"{metrics_text}\n\n"
            f"已选择: {'是' if selected_csv.exists() else '否'}"
        )
        self.set_status(str(csv_path))

    def metrics_text(self, csv_path: Path) -> str:
        if csv_path in self.metrics_cache:
            metrics = self.metrics_cache[csv_path]
        else:
            paths = self.region_paths()
            try:
                reference, predicted = collect_matches(csv_path, paths["cudem"], self.chunksize)
                if len(reference) < 3:
                    metrics = {"n": len(reference), "r2": np.nan, "mae": np.nan, "rmse": np.nan}
                else:
                    metrics = compute_metrics(reference, predicted)
            except Exception as exc:
                metrics = {"error": str(exc)}
            self.metrics_cache[csv_path] = metrics

        if "error" in metrics:
            return f"指标计算失败:\n{metrics['error']}"
        return (
            f"N = {metrics['n']}\n"
            f"R2 = {self.format_metric(metrics['r2'])}\n"
            f"MAE = {self.format_metric(metrics['mae'])} m\n"
            f"RMSE = {self.format_metric(metrics['rmse'])} m"
        )

    @staticmethod
    def format_metric(value):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.3f}"

    def refresh_current_item(self):
        if 0 <= self.current_index < self.file_list.count():
            csv_path = self.csv_paths[self.current_index]
            self.file_list.item(self.current_index).setText(self.list_label(csv_path))
            self.show_index(self.current_index)

    def select_current(self):
        csv_path = self.current_csv()
        if csv_path is None:
            return
        paths = self.region_paths()
        corr_png = paths["corr_dir"] / safe_output_name(csv_path)
        try:
            copied = copy_selected_files(csv_path, corr_png, paths["selected_dir"])
        except Exception as exc:
            QMessageBox.critical(self, "复制失败", str(exc))
            return
        self.refresh_current_item()
        self.set_status(f"Copied {len(copied)} files for {csv_path.name}")
        self.next_file()

    def next_file(self):
        if self.current_index + 1 < len(self.csv_paths):
            self.file_list.setCurrentRow(self.current_index + 1)

    def prev_file(self):
        if self.current_index > 0:
            self.file_list.setCurrentRow(self.current_index - 1)

    def open_selected_dir(self):
        selected_dir = self.region_paths()["selected_dir"]
        selected_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(selected_dir))

    def clear_selected_dir(self):
        selected_dir = self.region_paths()["selected_dir"]
        if not selected_dir.exists():
            self.set_status(f"Predictions_selected does not exist: {selected_dir}")
            return

        reply = QMessageBox.question(
            self,
            "确认清空",
            f"确定清空当前区域的 Predictions_selected 文件夹吗？\n\n{selected_dir}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        removed = 0
        failed = []
        for path in selected_dir.iterdir():
            if not path.is_file():
                continue
            try:
                path.unlink()
                removed += 1
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")

        self.reload_current_region()
        if failed:
            QMessageBox.warning(self, "部分文件未删除", "\n".join(failed[:20]))
        self.set_status(f"Removed {removed} files from {selected_dir}")

    def delete_current_group(self):
        csv_path = self.current_csv()
        if csv_path is None:
            return

        paths = self.region_paths()
        selected_dir = paths["selected_dir"]
        targets = [
            selected_dir / csv_path.name,
            selected_dir / f"{csv_path.stem}_result.png",
            selected_dir / f"{safe_output_name(csv_path)}",
        ]
        existing = [path for path in targets if path.exists()]
        if not existing:
            QMessageBox.information(
                self,
                "未找到文件",
                f"当前条目对应的文件组在 Predictions_selected 中不存在。\n\n{csv_path.name}",
            )
            return

        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定删除当前已复制的文件组吗？\n\n" + "\n".join(str(path) for path in existing),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        removed = 0
        failed = []
        for path in existing:
            try:
                path.unlink()
                removed += 1
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")

        self.refresh_current_item()
        if failed:
            QMessageBox.warning(self, "部分文件未删除", "\n".join(failed[:20]))
        self.set_status(f"Removed {removed} files for {csv_path.name}")

    def set_status(self, message: str):
        self.statusBar().showMessage(message)


def parse_args():
    parser = argparse.ArgumentParser(description="Manual Qt selector for ICESat-2 prediction files.")
    parser.add_argument("--region", choices=sorted(REGIONS), default=None, help="Initial region to display.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV rows read per chunk.")
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    window = ManualSelectionWindow(args.region, args.chunksize)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
