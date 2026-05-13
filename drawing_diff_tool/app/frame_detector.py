from __future__ import annotations

import cv2
import numpy as np

from drawing_diff_tool.app.models import FrameCandidate


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def candidate_from_rect(rect: tuple[tuple[float, float], tuple[float, float], float], score: float) -> FrameCandidate:
    box = cv2.boxPoints(rect)
    return FrameCandidate(points=order_points(box), score=score, source="auto")


def detect_frame_candidates(line_mask: np.ndarray, max_candidates: int = 5) -> list[FrameCandidate]:
    """Detect large rectangular frames near the page perimeter."""
    h, w = line_mask.shape[:2]
    page_area = float(h * w)
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[FrameCandidate] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < page_area * 0.05:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _angle = rect
        if rw <= 0 or rh <= 0:
            continue
        rect_area = float(rw * rh)
        if rect_area < page_area * 0.08:
            continue
        extent = min(1.0, contour_area / max(rect_area, 1.0))
        x, y, bw, bh = cv2.boundingRect(contour)
        edge_distance = min(x, y, w - (x + bw), h - (y + bh))
        edge_score = 1.0 - min(1.0, max(0, edge_distance) / max(w, h) * 8.0)
        area_score = min(1.0, rect_area / page_area)
        aspect_score = min(rw, rh) / max(rw, rh)
        score = area_score * 0.45 + edge_score * 0.35 + extent * 0.15 + aspect_score * 0.05
        candidates.append(candidate_from_rect(rect, score))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return _dedupe_candidates(candidates)[:max_candidates]


def _dedupe_candidates(candidates: list[FrameCandidate]) -> list[FrameCandidate]:
    unique: list[FrameCandidate] = []
    for candidate in candidates:
        keep = True
        for seen in unique:
            if np.linalg.norm(candidate.points.mean(axis=0) - seen.points.mean(axis=0)) < 10:
                if abs(candidate.width - seen.width) < 20 and abs(candidate.height - seen.height) < 20:
                    keep = False
                    break
        if keep:
            unique.append(candidate)
    return unique


def manual_candidate(points: list[tuple[float, float]]) -> FrameCandidate:
    if len(points) != 4:
        raise ValueError("Manual frame requires exactly four points")
    return FrameCandidate(points=order_points(np.asarray(points, dtype=np.float32)), score=1.0, source="manual")
