from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drawing_diff_tool.app.aligner import align_image_to_frame, overlay_preview
from drawing_diff_tool.app.change_region_detector import ChangeDetectionSettings, detect_change_regions
from drawing_diff_tool.app.diff_engine import DiffSettings, create_diff_image
from drawing_diff_tool.app.exporter import save_alignment_json, save_changes_csv, save_diff_pdf, save_png
from drawing_diff_tool.app.file_loader import LoadedDocument, open_document
from drawing_diff_tool.app.frame_detector import detect_frame_candidates, manual_candidate
from drawing_diff_tool.app.models import AlignmentParameters, FrameCandidate
from drawing_diff_tool.app.preprocess import PreprocessSettings, preprocess_image
from drawing_diff_tool.app.ui_viewer import ImageViewer, annotate_frames


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("図面差分・枠基準補正ツール")
        self.resize(1500, 950)

        self.old_doc: LoadedDocument | None = None
        self.new_doc: LoadedDocument | None = None
        self.old_image: np.ndarray | None = None
        self.new_image: np.ndarray | None = None
        self.aligned_new_image: np.ndarray | None = None
        self.diff_image: np.ndarray | None = None
        self.overlay_image: np.ndarray | None = None
        self.change_regions = []
        self.alignment_params: AlignmentParameters | None = None
        self.old_frames: list[FrameCandidate] = []
        self.new_frames: list[FrameCandidate] = []
        self.selected_old_frame: int | None = None
        self.selected_new_frame: int | None = None
        self._manual_target: str | None = None
        self._manual_points: list[tuple[float, float]] = []

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("操作順: 1.旧図面を選択 → 2.新図面を選択 → 3.枠を検出 → 4.枠を選択 → 5.補正 → 6.差分表示 → 7.出力")
        title.setStyleSheet("font-weight: bold; padding: 6px; background: #eef6ff;")
        layout.addWidget(title)

        controls = QGridLayout()
        layout.addLayout(controls)
        self.old_label = QLabel("未選択")
        self.new_label = QLabel("未選択")
        old_btn = QPushButton("1. 旧図面を選択"); old_btn.clicked.connect(self.select_old_file)
        controls.addWidget(old_btn, 0, 0)
        controls.addWidget(self.old_label, 0, 1)
        new_btn = QPushButton("2. 新図面を選択"); new_btn.clicked.connect(self.select_new_file)
        controls.addWidget(new_btn, 0, 2)
        controls.addWidget(self.new_label, 0, 3)

        self.page_combo = QComboBox()
        self.page_combo.currentIndexChanged.connect(self._load_selected_page)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(400)
        self.dpi_spin.setSuffix(" dpi")
        controls.addWidget(QLabel("ページ"), 1, 0)
        controls.addWidget(self.page_combo, 1, 1)
        controls.addWidget(QLabel("DPI"), 1, 2)
        controls.addWidget(self.dpi_spin, 1, 3)

        settings_group = QGroupBox("差分・検出設定")
        form = QFormLayout(settings_group)
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["otsu", "fixed", "adaptive"])
        self.fixed_threshold = QSpinBox(); self.fixed_threshold.setRange(0, 255); self.fixed_threshold.setValue(180)
        self.tolerance_spin = QSpinBox(); self.tolerance_spin.setRange(0, 5); self.tolerance_spin.setValue(2)
        self.denoise_spin = QSpinBox(); self.denoise_spin.setRange(0, 10000); self.denoise_spin.setValue(12)
        self.merge_spin = QSpinBox(); self.merge_spin.setRange(0, 500); self.merge_spin.setValue(25)
        self.margin_spin = QSpinBox(); self.margin_spin.setRange(0, 200); self.margin_spin.setValue(12)
        form.addRow("二値化方式", self.threshold_combo)
        form.addRow("固定しきい値", self.fixed_threshold)
        form.addRow("線幅許容(px)", self.tolerance_spin)
        form.addRow("ノイズ除去面積(px)", self.denoise_spin)
        form.addRow("変更箇所結合距離(px)", self.merge_spin)
        form.addRow("変更範囲余白(px)", self.margin_spin)
        controls.addWidget(settings_group, 2, 0, 1, 4)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)
        for text, slot in [
            ("3. 外枠自動検出", self.detect_frames),
            ("旧枠を手動四隅指定", lambda: self.start_manual_frame("old")),
            ("新枠を手動四隅指定", lambda: self.start_manual_frame("new")),
            ("5. 補正プレビュー", self.preview_alignment),
            ("6. 差分実行", self.run_diff),
            ("PNG出力", self.export_png),
            ("PDF出力", self.export_pdf),
            ("CSV出力", self.export_csv),
            ("JSON出力", self.export_json),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _checked=False, callback=slot: callback())
            button_row.addWidget(btn)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.old_viewer = ImageViewer(); self.new_viewer = ImageViewer(); self.diff_viewer = ImageViewer(); self.overlay_viewer = ImageViewer()
        self.old_viewer.clicked.connect(lambda x, y: self.handle_view_click("old", x, y))
        self.new_viewer.clicked.connect(lambda x, y: self.handle_view_click("new", x, y))
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._labeled("旧図面", self.old_viewer))
        split.addWidget(self._labeled("新図面", self.new_viewer))
        split.addWidget(self._labeled("差分結果", self.diff_viewer))
        self.tabs.addTab(split, "比較")
        self.tabs.addTab(self._labeled("重ね合わせ", self.overlay_viewer), "重ね合わせ")
        self.statusBar().showMessage("旧図面と新図面を選択してください。")

    def _labeled(self, label: str, widget: QWidget) -> QWidget:
        box = QWidget(); layout = QVBoxLayout(box); layout.addWidget(QLabel(label)); layout.addWidget(widget, 1); return box

    def select_old_file(self) -> None:
        self._select_file("old")

    def select_new_file(self) -> None:
        self._select_file("new")

    def _select_file(self, kind: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "図面ファイルを選択", "", "Drawings (*.pdf *.tif *.tiff *.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            doc = open_document(path, self.dpi_spin.value())
        except Exception as exc:
            QMessageBox.critical(self, "読込エラー", str(exc)); return
        if kind == "old":
            self.old_doc = doc; self.old_label.setText(Path(path).name)
        else:
            self.new_doc = doc; self.new_label.setText(Path(path).name)
        self._refresh_pages()

    def _refresh_pages(self) -> None:
        self.page_combo.blockSignals(True)
        self.page_combo.clear()
        counts = [d.page_count for d in (self.old_doc, self.new_doc) if d]
        for i in range(min(counts) if counts else 0):
            self.page_combo.addItem(f"{i + 1} ページ", i)
        self.page_combo.blockSignals(False)
        self._load_selected_page()

    def _load_selected_page(self) -> None:
        index = self.page_combo.currentData()
        if index is None or not (self.old_doc and self.new_doc):
            return
        self.old_image = self.old_doc.page(index)
        self.new_image = self.new_doc.page(index)
        self.aligned_new_image = None; self.diff_image = None; self.change_regions = []
        self.old_frames = []; self.new_frames = []; self.selected_old_frame = None; self.selected_new_frame = None
        self.old_viewer.set_image(self.old_image); self.new_viewer.set_image(self.new_image); self.diff_viewer.set_image(None); self.overlay_viewer.set_image(None)

    def _preprocess_settings(self) -> PreprocessSettings:
        return PreprocessSettings(
            threshold_mode=self.threshold_combo.currentText(),
            fixed_threshold=self.fixed_threshold.value(),
            denoise_area_px=self.denoise_spin.value(),
            line_dilate_px=1,
        )

    def detect_frames(self) -> None:
        if self.old_image is None or self.new_image is None:
            QMessageBox.warning(self, "未選択", "旧図面と新図面を先に選択してください。"); return
        settings = self._preprocess_settings()
        self.old_frames = detect_frame_candidates(preprocess_image(self.old_image, settings))
        self.new_frames = detect_frame_candidates(preprocess_image(self.new_image, settings))
        self.selected_old_frame = 0 if self.old_frames else None
        self.selected_new_frame = 0 if self.new_frames else None
        self._update_frame_views()
        self.statusBar().showMessage("枠候補をクリックして基準枠を選択してください。候補がない場合は手動指定を使ってください。")

    def _update_frame_views(self) -> None:
        self.old_viewer.set_image(annotate_frames(self.old_image, self.old_frames, self.selected_old_frame) if self.old_image is not None else None)
        self.new_viewer.set_image(annotate_frames(self.new_image, self.new_frames, self.selected_new_frame) if self.new_image is not None else None)

    def handle_view_click(self, kind: str, x: float, y: float) -> None:
        if self._manual_target == kind:
            self._manual_points.append((x, y))
            self.statusBar().showMessage(f"{kind} 手動枠: {len(self._manual_points)}/4 点")
            if len(self._manual_points) == 4:
                frame = manual_candidate(self._manual_points)
                if kind == "old":
                    self.old_frames.append(frame); self.selected_old_frame = len(self.old_frames) - 1
                else:
                    self.new_frames.append(frame); self.selected_new_frame = len(self.new_frames) - 1
                self._manual_target = None; self._manual_points = []; self._update_frame_views()
            return
        frames = self.old_frames if kind == "old" else self.new_frames
        if not frames:
            return
        point = np.array([x, y], dtype=np.float32)
        distances = [cv2.pointPolygonTest(f.points.astype(np.float32), (float(point[0]), float(point[1])), True) for f in frames]
        best = int(np.argmax(distances))
        if distances[best] >= -20:
            if kind == "old": self.selected_old_frame = best
            else: self.selected_new_frame = best
            self._update_frame_views()

    def start_manual_frame(self, kind: str) -> None:
        self._manual_target = kind; self._manual_points = []
        self.statusBar().showMessage(("旧" if kind == "old" else "新") + "図面ビューで枠の四隅をクリックしてください。")

    def preview_alignment(self) -> None:
        if not self._ensure_frames(): return
        src = self.new_frames[self.selected_new_frame]
        dst = self.old_frames[self.selected_old_frame]
        self.aligned_new_image, self.alignment_params, _ = align_image_to_frame(
            self.new_image, src, dst, self.dpi_spin.value(), self.old_image.shape, method="similarity"
        )
        self.overlay_image = overlay_preview(self.old_image, self.aligned_new_image)
        self.new_viewer.set_image(self.aligned_new_image); self.overlay_viewer.set_image(self.overlay_image); self.tabs.setCurrentIndex(1)

    def run_diff(self) -> None:
        if self.old_image is None or self.new_image is None:
            QMessageBox.warning(self, "未選択", "旧図面と新図面を先に選択してください。"); return
        if self.aligned_new_image is None:
            if self.selected_old_frame is not None and self.selected_new_frame is not None:
                self.preview_alignment()
            else:
                self.aligned_new_image = cv2.resize(self.new_image, (self.old_image.shape[1], self.old_image.shape[0]))
        settings = self._preprocess_settings()
        old_mask = preprocess_image(self.old_image, settings)
        new_mask = preprocess_image(self.aligned_new_image, settings)
        self.diff_image, masks = create_diff_image(old_mask, new_mask, DiffSettings(self.tolerance_spin.value()))
        page_no = (self.page_combo.currentIndex() + 1) if self.page_combo.currentIndex() >= 0 else 1
        self.change_regions = detect_change_regions(
            masks,
            page=page_no,
            settings=ChangeDetectionSettings(self.merge_spin.value(), self.denoise_spin.value(), self.margin_spin.value()),
        )
        self.diff_viewer.set_image(self.diff_image)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(f"差分完了: 変更箇所 {len(self.change_regions)} 件")

    def _ensure_frames(self) -> bool:
        if self.old_image is None or self.new_image is None or self.selected_old_frame is None or self.selected_new_frame is None:
            QMessageBox.warning(self, "枠未選択", "外枠を検出し、旧図面・新図面の基準枠を選択してください。"); return False
        return True

    def export_png(self) -> None:
        if self.diff_image is None: self.run_diff()
        if self.diff_image is None: return
        path, _ = QFileDialog.getSaveFileName(self, "PNG出力", "diff.png", "PNG (*.png)")
        if path: save_png(path, self.diff_image, self.change_regions)

    def export_pdf(self) -> None:
        if self.diff_image is None: self.run_diff()
        if self.diff_image is None: return
        path, _ = QFileDialog.getSaveFileName(self, "PDF出力", "diff.pdf", "PDF (*.pdf)")
        if path: save_diff_pdf(path, [self.diff_image], [self.change_regions], self.dpi_spin.value())

    def export_csv(self) -> None:
        if self.diff_image is None: self.run_diff()
        path, _ = QFileDialog.getSaveFileName(self, "CSV出力", "changes.csv", "CSV (*.csv)")
        if path: save_changes_csv(path, self.change_regions)

    def export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "JSON出力", "alignment.json", "JSON (*.json)")
        if path:
            save_alignment_json(path, self.alignment_params, {"changes": [r.to_csv_row() for r in self.change_regions]})
