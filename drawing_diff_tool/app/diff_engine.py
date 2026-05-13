from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class DiffSettings:
    tolerance_px: int = 2


@dataclass(slots=True)
class DiffMasks:
    common: np.ndarray
    old_only: np.ndarray
    new_only: np.ndarray


def compare_line_masks(old_mask: np.ndarray, new_mask: np.ndarray, settings: DiffSettings | None = None) -> DiffMasks:
    settings = settings or DiffSettings()
    old = (old_mask > 0).astype(np.uint8) * 255
    new = (new_mask > 0).astype(np.uint8) * 255
    if old.shape != new.shape:
        new = cv2.resize(new, (old.shape[1], old.shape[0]), interpolation=cv2.INTER_NEAREST)
    if settings.tolerance_px > 0:
        size = settings.tolerance_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        old_tol = cv2.dilate(old, kernel)
        new_tol = cv2.dilate(new, kernel)
    else:
        old_tol = old
        new_tol = new
    common = cv2.bitwise_and(old, new_tol)
    common = cv2.bitwise_or(common, cv2.bitwise_and(new, old_tol))
    old_only = cv2.bitwise_and(old, cv2.bitwise_not(new_tol))
    new_only = cv2.bitwise_and(new, cv2.bitwise_not(old_tol))
    return DiffMasks(common=common, old_only=old_only, new_only=new_only)


def render_diff(masks: DiffMasks) -> np.ndarray:
    h, w = masks.common.shape[:2]
    output = np.full((h, w, 3), 255, dtype=np.uint8)
    output[masks.common > 0] = (0, 0, 0)
    output[masks.old_only > 0] = (0, 0, 255)
    output[masks.new_only > 0] = (255, 0, 0)
    return output


def create_diff_image(old_mask: np.ndarray, new_mask: np.ndarray, settings: DiffSettings | None = None) -> tuple[np.ndarray, DiffMasks]:
    masks = compare_line_masks(old_mask, new_mask, settings)
    return render_diff(masks), masks
