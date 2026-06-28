import argparse
import os
import shutil
import sys
from pathlib import Path
    
import numpy as np
import rasterio
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QPixmap, QImage
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
BASE_DIR = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data")
AREAS = ["Florida Bay", "Key Largo", "Key West", "Marathon"]
S2_SUBDIR = Path("Sentinel-2")
TIF_FINAL_DIR_NAME = "TIF_Final"
BACKUP_DIR_NAME = "TIF_Final_Removed"
TCI_FILE = "TCI.tif"
REQUIRED_FILES = (
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B08.tif",
    "NDWI.tif",
    "Blue_Green_LogRatio.tif",
    "Blue_Red_LogRatio.tif",
    "Green_Red_LogRatio.tif",
    "cloud_mask.tif",
)


def stretch_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)

    values = arr[valid]
    lo = np.percentile(values, 2)
    hi = np.percentile(values, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~valid] = 0.0
    return (scaled * 255.0).astype(np.uint8)


def list_scenes(tif_dir: Path) -> list[Path]:
    if not tif_dir.exists():
        return []
    return sorted(p for p in tif_dir.iterdir() if p.is_dir())


def scene_status(scene_dir: Path) -> tuple[str, str]:
    required = [scene_dir / name for name in REQUIRED_FILES]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        return "不完整", f"缺少: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}"
    return "完整", "OK"


def unique_backup_path(target_dir: Path, scene_name: str) -> Path:
    candidate = target_dir / scene_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = target_dir / f"{scene_name}__removed_{suffix:03d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def scene_preview(scene_dir: Path) -> QPixmap | None:
    tci_path = scene_dir / TCI_FILE
    if not tci_path.exists():
        return None

    try:
        with rasterio.open(tci_path) as src:
            if src.count < 3:
                return None
            out_height = max(1, src.height // 4)
            out_width = max(1, src.width // 4)
            bands = src.read(
                indexes=[1, 2, 3],
                out_shape=(3, out_height, out_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )

        r = stretch_to_uint8(bands[0])
        g = stretch_to_uint8(bands[1])
        b = stretch_to_uint8(bands[2])
        rgb = np.dstack([r, g, b])
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)
    except Exception:
        return None


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("No preview")

    def set_pixmap(self, pixmap: QPixmap | None, text: str = "No preview"):
        self._pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(text)
        else:
            self._apply_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled()

    def _apply_scaled(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class TifFinalSelector(QMainWindow):
    def __init__(self, initial_region: str | None):
        super().__init__()
        self.current_region = None
        self.scene_dirs: list[Path] = []
        self.current_index = -1

        self.setWindowTitle("Sentinel-2 TIF_Final Manual Selector")
        self.resize(1400, 860)

        self.region_box = QComboBox()
        self.region_box.addItems(AREAS)
        if initial_region in AREAS:
            self.region_box.setCurrentText(initial_region)
        self.region_box.currentTextChanged.connect(self.load_region)

        self.scene_list = QListWidget()
        self.scene_list.currentRowChanged.connect(self.show_scene)

        self.preview = ImageLabel()
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setMinimumWidth(360)

        self.keep_button = QPushButton("保留")
        self.remove_button = QPushButton("删除并移至备份")
        self.prev_button = QPushButton("上一景")
        self.next_button = QPushButton("下一景")
        self.refresh_button = QPushButton("刷新")
        self.open_source_button = QPushButton("打开源文件夹")
        self.open_backup_button = QPushButton("打开备份目录")

        self.keep_button.clicked.connect(self.keep_scene)
        self.remove_button.clicked.connect(self.remove_scene)
        self.prev_button.clicked.connect(self.prev_scene)
        self.next_button.clicked.connect(self.next_scene)
        self.refresh_button.clicked.connect(self.reload_current_region)
        self.open_source_button.clicked.connect(self.open_source_dir)
        self.open_backup_button.clicked.connect(self.open_backup_dir)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("区域"))
        left_layout.addWidget(self.region_box)
        left_layout.addWidget(QLabel("场景列表"))
        left_layout.addWidget(self.scene_list)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(QLabel("TCI 预览"))
        center_layout.addWidget(self.preview)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("信息"))
        right_layout.addWidget(self.info_label)
        right_layout.addStretch(1)
        right_layout.addWidget(self.keep_button)
        right_layout.addWidget(self.remove_button)
        right_layout.addWidget(self.prev_button)
        right_layout.addWidget(self.next_button)
        right_layout.addWidget(self.refresh_button)
        right_layout.addWidget(self.open_source_button)
        right_layout.addWidget(self.open_backup_button)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([300, 760, 340])

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

        actions = [
            ("Keep", QKeySequence(Qt.Key_A), self.keep_scene),
            ("Remove", QKeySequence(Qt.Key_D), self.remove_scene),
            ("Prev", QKeySequence(Qt.Key_Left), self.prev_scene),
            ("Next", QKeySequence(Qt.Key_Right), self.next_scene),
            ("EnterKeep", QKeySequence(Qt.Key_Return), self.keep_scene),
        ]
        for name, sequence, slot in actions:
            action = QAction(name, self)
            action.setShortcut(sequence)
            action.triggered.connect(slot)
            toolbar.addAction(action)

    def paths(self):
        root = BASE_DIR / self.current_region
        tif_dir = root / S2_SUBDIR / TIF_FINAL_DIR_NAME
        backup_dir = root / S2_SUBDIR / BACKUP_DIR_NAME
        return root, tif_dir, backup_dir

    def load_region(self, region_name: str, target_row: int | None = 0):
        self.current_region = region_name
        self.scene_dirs = []
        self.current_index = -1
        self.scene_list.clear()

        _, tif_dir, _ = self.paths()
        if not tif_dir.exists():
            self.preview.set_pixmap(None, text=f"Missing folder:\n{tif_dir}")
            self.info_label.setText("TIF_Final folder not found.")
            self.set_status(f"Missing: {tif_dir}")
            return

        self.scene_dirs = list_scenes(tif_dir)
        for scene_dir in self.scene_dirs:
            status, detail = scene_status(scene_dir)
            item = QListWidgetItem(f"[{status}] {scene_dir.name}")
            item.setToolTip(f"{scene_dir}\n{detail}")
            self.scene_list.addItem(item)

        if self.scene_dirs:
            if target_row is None:
                target_row = 0
            target_row = max(0, min(target_row, len(self.scene_dirs) - 1))
            self.scene_list.setCurrentRow(target_row)
        else:
            self.preview.set_pixmap(None, text="No scenes found")
            self.info_label.setText("No scenes found.")

        self.set_status(f"{region_name}: {len(self.scene_dirs)} scenes")

    def reload_current_region(self):
        current_scene = self.current_scene()
        self.load_region(self.current_region, target_row=None)
        if current_scene is not None and current_scene in self.scene_dirs:
            self.scene_list.setCurrentRow(self.scene_dirs.index(current_scene))

    def current_scene(self) -> Path | None:
        if 0 <= self.current_index < len(self.scene_dirs):
            return self.scene_dirs[self.current_index]
        return None

    def show_scene(self, row: int):
        self.current_index = row
        scene_dir = self.current_scene()
        if scene_dir is None:
            return

        root, _, backup_dir = self.paths()
        status, detail = scene_status(scene_dir)
        pixmap = scene_preview(scene_dir)
        if pixmap is None:
            self.preview.set_pixmap(None, text="TCI preview unavailable")
        else:
            self.preview.set_pixmap(pixmap)

        rel_path = scene_dir.relative_to(root)
        self.info_label.setText(
            f"区域: {self.current_region}\n"
            f"序号: {row + 1} / {len(self.scene_dirs)}\n\n"
            f"场景: {scene_dir.name}\n"
            f"路径: {scene_dir}\n\n"
            f"状态: {status}\n"
            f"说明: {detail}\n\n"
            f"源目录: {root / S2_SUBDIR / TIF_FINAL_DIR_NAME}\n"
            f"备份目录: {backup_dir}\n"
            f"相对路径: {rel_path}"
        )
        self.set_status(str(scene_dir))

    def keep_scene(self):
        scene_dir = self.current_scene()
        if scene_dir is None:
            return
        self.set_status(f"Kept {scene_dir.name}")
        self.next_scene()

    def remove_scene(self):
        scene_dir = self.current_scene()
        if scene_dir is None:
            return

        root, _, backup_dir = self.paths()
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = unique_backup_path(backup_dir, scene_dir.name)

        reply = QMessageBox.question(
            self,
            "确认移动",
            f"确定将该场景移至备份目录吗？\n\n{scene_dir}\n\n->\n\n{target}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            shutil.move(str(scene_dir), str(target))
        except Exception as exc:
            QMessageBox.critical(self, "移动失败", str(exc))
            return

        target_row = self.current_index
        self.load_region(self.current_region, target_row=target_row)
        self.set_status(f"Moved to backup: {target}")

    def prev_scene(self):
        if self.current_index > 0:
            self.scene_list.setCurrentRow(self.current_index - 1)

    def next_scene(self):
        if self.current_index + 1 < len(self.scene_dirs):
            self.scene_list.setCurrentRow(self.current_index + 1)

    def open_source_dir(self):
        _, tif_dir, _ = self.paths()
        tif_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(tif_dir))

    def open_backup_dir(self):
        _, _, backup_dir = self.paths()
        backup_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(backup_dir))

    def set_status(self, message: str):
        self.statusBar().showMessage(message)


def parse_args():
    parser = argparse.ArgumentParser(description="Manual Qt selector for Sentinel-2 TIF_Final scenes.")
    parser.add_argument("--region", choices=AREAS, default=None, help="Initial region to display.")
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    window = TifFinalSelector(args.region)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
