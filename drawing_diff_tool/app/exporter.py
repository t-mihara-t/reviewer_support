from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from drawing_diff_tool.app.change_region_detector import draw_change_regions
from drawing_diff_tool.app.models import AlignmentParameters, ChangeRegion


def save_png(path: str | Path, image: np.ndarray, regions: list[ChangeRegion] | None = None) -> None:
    output = draw_change_regions(image, regions or []) if regions else image
    bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Failed to write PNG: {path}")


def save_changes_csv(path: str | Path, regions: list[ChangeRegion]) -> None:
    fieldnames = ["page", "change_no", "x", "y", "width", "height", "type", "area_px"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for region in regions:
            writer.writerow(region.to_csv_row())


def save_alignment_json(
    path: str | Path,
    params: AlignmentParameters | None,
    extra: dict | None = None,
) -> None:
    payload = {"alignment": params.to_json() if params else None}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_diff_pdf(
    path: str | Path,
    images: list[np.ndarray],
    regions_by_page: list[list[ChangeRegion]] | None = None,
    dpi: int = 400,
) -> None:
    regions_by_page = regions_by_page or [[] for _ in images]
    pdf = canvas.Canvas(str(path))
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, image in enumerate(images):
            annotated = draw_change_regions(image, regions_by_page[index] if index < len(regions_by_page) else [])
            tmp_path = Path(tmpdir) / f"page_{index + 1}.png"
            cv2.imwrite(str(tmp_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            h, w = annotated.shape[:2]
            page_w = w * 72.0 / dpi
            page_h = h * 72.0 / dpi
            pdf.setPageSize((page_w, page_h))
            pdf.drawImage(ImageReader(str(tmp_path)), 0, 0, width=page_w, height=page_h)
            pdf.showPage()
    pdf.save()
