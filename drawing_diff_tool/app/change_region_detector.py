from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drawing_diff_tool.app.diff_engine import DiffMasks
from drawing_diff_tool.app.models import ChangeRegion


@dataclass(slots=True)
class ChangeDetectionSettings:
    merge_distance_px: int = 25
    min_area_px: int = 50
    margin_px: int = 12


def detect_change_regions(
    masks: DiffMasks,
    page: int = 1,
    settings: ChangeDetectionSettings | None = None,
) -> list[ChangeRegion]:
    settings = settings or ChangeDetectionSettings()
    old_only = (masks.old_only > 0).astype(np.uint8) * 255
    new_only = (masks.new_only > 0).astype(np.uint8) * 255
    diff = cv2.bitwise_or(old_only, new_only)
    if settings.merge_distance_px > 0:
        k = settings.merge_distance_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        diff = cv2.dilate(diff, kernel, iterations=1)
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = diff.shape[:2]
    regions: list[ChangeRegion] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        x0 = max(0, x - settings.margin_px)
        y0 = max(0, y - settings.margin_px)
        x1 = min(w, x + bw + settings.margin_px)
        y1 = min(h, y + bh + settings.margin_px)
        old_area = int(np.count_nonzero(old_only[y0:y1, x0:x1]))
        new_area = int(np.count_nonzero(new_only[y0:y1, x0:x1]))
        area = old_area + new_area
        if area < settings.min_area_px:
            continue
        if old_area and new_area:
            change_type = "mixed"
        elif old_area:
            change_type = "old_only"
        else:
            change_type = "new_only"
        regions.append(
            ChangeRegion(
                page=page,
                change_no=0,
                x=int(x0),
                y=int(y0),
                width=int(x1 - x0),
                height=int(y1 - y0),
                type=change_type,
                area_px=area,
            )
        )
    regions.sort(key=lambda r: (r.y, r.x))
    for index, region in enumerate(regions, start=1):
        region.change_no = index
    return regions


def draw_change_regions(image: np.ndarray, regions: list[ChangeRegion]) -> np.ndarray:
    output = image.copy()
    for region in regions:
        color = (255, 0, 255) if region.type == "mixed" else ((0, 0, 255) if region.type == "old_only" else (255, 0, 0))
        cv2.rectangle(output, (region.x, region.y), (region.x + region.width, region.y + region.height), color, 3)
        cv2.putText(
            output,
            str(region.change_no),
            (region.x + 6, max(20, region.y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
