from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from drawing_diff_tool.app.change_region_detector import ChangeDetectionSettings
from drawing_diff_tool.app.diff_engine import DiffSettings
from drawing_diff_tool.app.preprocess import PreprocessSettings


@dataclass(slots=True)
class AppSettings:
    dpi: int = 400
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)
    diff: DiffSettings = field(default_factory=DiffSettings)
    change_detection: ChangeDetectionSettings = field(default_factory=ChangeDetectionSettings)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "AppSettings":
        return cls(
            dpi=int(data.get("dpi", 400)),
            preprocess=PreprocessSettings(**data.get("preprocess", {})),
            diff=DiffSettings(**data.get("diff", {})),
            change_detection=ChangeDetectionSettings(**data.get("change_detection", {})),
        )


def load_settings(path: str | Path) -> AppSettings:
    with open(path, encoding="utf-8") as f:
        return AppSettings.from_json(json.load(f))


def save_settings(path: str | Path, settings: AppSettings) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings.to_json(), f, ensure_ascii=False, indent=2)
