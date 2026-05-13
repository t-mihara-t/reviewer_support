from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

ThresholdMode = Literal["fixed", "otsu", "adaptive"]


@dataclass(slots=True)
class PreprocessSettings:
    threshold_mode: ThresholdMode = "otsu"
    fixed_threshold: int = 180
    denoise_area_px: int = 12
    line_dilate_px: int = 1
    adaptive_block_size: int = 35
    adaptive_c: int = 11


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.astype(np.uint8, copy=False)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    rgb = ensure_uint8_rgb(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def threshold_lines(gray: np.ndarray, settings: PreprocessSettings) -> np.ndarray:
    """Return a binary mask where drawing lines are 255 and background is 0."""
    gray = gray.astype(np.uint8, copy=False)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if settings.threshold_mode == "fixed":
        _, binary_inv = cv2.threshold(
            normalized, settings.fixed_threshold, 255, cv2.THRESH_BINARY_INV
        )
    elif settings.threshold_mode == "adaptive":
        block = max(3, settings.adaptive_block_size | 1)
        binary_inv = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            settings.adaptive_c,
        )
    else:
        _, binary_inv = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
    return clean_line_mask(binary_inv, settings.denoise_area_px, settings.line_dilate_px)


def clean_line_mask(mask: np.ndarray, denoise_area_px: int = 12, dilate_px: int = 1) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    if denoise_area_px > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= denoise_area_px:
                cleaned[labels == label] = 255
        mask = cleaned
    if dilate_px > 0:
        size = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def preprocess_image(image: np.ndarray, settings: PreprocessSettings | None = None) -> np.ndarray:
    return threshold_lines(to_gray(image), settings or PreprocessSettings())
