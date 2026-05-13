from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drawing_diff_tool.app.rasterizer import load_pages


@dataclass(slots=True)
class LoadedDocument:
    path: Path
    pages: list[np.ndarray]
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.page_count:
            raise IndexError(f"Page index out of range: {index}")
        return self.pages[index]


def open_document(path: str | Path, dpi: int = 400) -> LoadedDocument:
    pages, metadata = load_pages(path, dpi=dpi)
    return LoadedDocument(path=Path(path), pages=pages, metadata=metadata)
