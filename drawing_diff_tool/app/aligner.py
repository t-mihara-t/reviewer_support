from __future__ import annotations

import cv2
import numpy as np

from drawing_diff_tool.app.models import AlignmentParameters, FrameCandidate


def align_image_to_frame(
    moving_image: np.ndarray,
    source_frame: FrameCandidate,
    target_frame: FrameCandidate,
    dpi: int,
    output_shape: tuple[int, int],
    method: str = "similarity",
) -> tuple[np.ndarray, AlignmentParameters, np.ndarray]:
    """Warp moving_image so source_frame aligns to target_frame."""
    dst_h, dst_w = output_shape[:2]
    if method == "perspective":
        matrix = cv2.getPerspectiveTransform(
            source_frame.points.astype(np.float32), target_frame.points.astype(np.float32)
        )
        warped = cv2.warpPerspective(
            moving_image, matrix, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
        )
        params = _parameters_from_frames(source_frame, target_frame, dpi, "perspective")
        return warped, params, matrix

    matrix, _ = cv2.estimateAffinePartial2D(
        source_frame.points.astype(np.float32),
        target_frame.points.astype(np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        matrix = _fallback_affine(source_frame, target_frame)
    warped = cv2.warpAffine(
        moving_image, matrix, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    params = _parameters_from_affine(matrix, dpi)
    return warped, params, matrix


def _fallback_affine(source: FrameCandidate, target: FrameCandidate) -> np.ndarray:
    sx = target.width / max(source.width, 1e-6)
    sy = target.height / max(source.height, 1e-6)
    sc = source.center
    tc = target.center
    angle = np.radians(target.angle_deg - source.angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    linear = np.array([[sx * cos_a, -sy * sin_a], [sx * sin_a, sy * cos_a]], dtype=np.float32)
    src_center = np.array(sc, dtype=np.float32)
    dst_center = np.array(tc, dtype=np.float32)
    translation = dst_center - linear @ src_center
    return np.column_stack([linear, translation]).astype(np.float32)


def _parameters_from_affine(matrix: np.ndarray, dpi: int) -> AlignmentParameters:
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    scale_x = float(np.sqrt(a * a + c * c))
    scale_y = float(np.sqrt(b * b + d * d))
    rotation_deg = float(np.degrees(np.arctan2(c, a)))
    return AlignmentParameters(
        dpi=dpi,
        translation_x_px=float(tx),
        translation_y_px=float(ty),
        scale_x=scale_x,
        scale_y=scale_y,
        rotation_deg=rotation_deg,
        method="similarity",
    )


def _parameters_from_frames(source: FrameCandidate, target: FrameCandidate, dpi: int, method: str) -> AlignmentParameters:
    sx = target.width / max(source.width, 1e-6)
    sy = target.height / max(source.height, 1e-6)
    sc = source.center
    tc = target.center
    return AlignmentParameters(
        dpi=dpi,
        translation_x_px=tc[0] - sc[0],
        translation_y_px=tc[1] - sc[1],
        scale_x=sx,
        scale_y=sy,
        rotation_deg=target.angle_deg - source.angle_deg,
        method=method,
    )


def overlay_preview(old_image: np.ndarray, aligned_new_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    old = _ensure_rgb(old_image)
    new = _ensure_rgb(aligned_new_image)
    if old.shape != new.shape:
        new = cv2.resize(new, (old.shape[1], old.shape[0]))
    return cv2.addWeighted(old, alpha, new, 1.0 - alpha, 0)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image
