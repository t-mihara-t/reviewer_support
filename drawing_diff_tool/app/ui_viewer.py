from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from drawing_diff_tool.app.models import FrameCandidate


def ndarray_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image[:, :, :3].astype(np.uint8, copy=False)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def annotate_frames(image: np.ndarray, frames: list[FrameCandidate], selected_index: int | None = None) -> np.ndarray:
    output = image.copy()
    colors = [(0, 180, 0), (255, 128, 0), (128, 0, 255), (0, 180, 255), (255, 0, 128)]
    for i, frame in enumerate(frames):
        color = (255, 0, 255) if selected_index == i else colors[i % len(colors)]
        pts = frame.points.astype(int).reshape(-1, 1, 2)
        cv2.polylines(output, [pts], True, color, 3)
        x, y = pts[0, 0]
        cv2.putText(output, str(i + 1), (int(x) + 8, int(y) + 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return output


class ImageViewer(QGraphicsView):
    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.lightGray)

    def set_image(self, image: np.ndarray | None) -> None:
        self._scene.clear()
        self._pixmap_item = None
        if image is None:
            return
        pixmap = QPixmap.fromImage(ndarray_to_qimage(image))
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            pos = self.mapToScene(event.pos())
            self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)

    def zoom_in(self) -> None:
        self.scale(1.25, 1.25)

    def zoom_out(self) -> None:
        self.scale(0.8, 0.8)

    def reset_zoom(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
