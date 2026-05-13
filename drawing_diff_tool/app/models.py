from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


FrameSource = Literal["auto", "manual"]
ChangeType = Literal["old_only", "new_only", "mixed"]


@dataclass(slots=True)
class FrameCandidate:
    """A detected or user-provided drawing frame."""

    points: np.ndarray  # shape=(4, 2), ordered tl,tr,br,bl
    score: float = 0.0
    source: FrameSource = "auto"

    @property
    def x(self) -> float:
        return float(np.min(self.points[:, 0]))

    @property
    def y(self) -> float:
        return float(np.min(self.points[:, 1]))

    @property
    def width(self) -> float:
        return float(np.max(self.points[:, 0]) - self.x)

    @property
    def height(self) -> float:
        return float(np.max(self.points[:, 1]) - self.y)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def angle_deg(self) -> float:
        tl, tr = self.points[0], self.points[1]
        return float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))

    def to_json(self) -> dict:
        return {
            "points": self.points.astype(float).round(3).tolist(),
            "score": round(float(self.score), 6),
            "source": self.source,
        }


@dataclass(slots=True)
class AlignmentParameters:
    dpi: int
    translation_x_px: float
    translation_y_px: float
    scale_x: float
    scale_y: float
    rotation_deg: float
    method: str = "similarity"

    def to_json(self) -> dict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 6)
        return data


@dataclass(slots=True)
class ChangeRegion:
    page: int
    change_no: int
    x: int
    y: int
    width: int
    height: int
    type: ChangeType
    area_px: int

    def to_csv_row(self) -> dict:
        return asdict(self)
