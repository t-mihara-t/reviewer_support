from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageSequence

SUPPORTED_EXTENSIONS = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def rasterize_pdf(path: str | Path, dpi: int = 400) -> list[np.ndarray]:
    doc = fitz.open(str(path))
    pages: list[np.ndarray] = []
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            pages.append(arr[:, :, :3].copy())
    finally:
        doc.close()
    return pages


def load_bitmap_pages(path: str | Path) -> tuple[list[np.ndarray], tuple[float, float] | None]:
    image = Image.open(path)
    dpi = image.info.get("dpi")
    pages: list[np.ndarray] = []
    for frame in ImageSequence.Iterator(image):
        rgb = frame.convert("RGB")
        pages.append(np.asarray(rgb).copy())
    return pages, dpi


def load_pages(path: str | Path, dpi: int = 400) -> tuple[list[np.ndarray], dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")
    if suffix == ".pdf":
        pages = rasterize_pdf(path, dpi)
        metadata = {"path": str(path), "kind": "pdf", "dpi": dpi, "page_count": len(pages)}
    else:
        pages, embedded_dpi = load_bitmap_pages(path)
        metadata = {
            "path": str(path),
            "kind": suffix.lstrip("."),
            "dpi": embedded_dpi or (dpi, dpi),
            "page_count": len(pages),
        }
    if not pages:
        raise ValueError(f"No pages found in {path}")
    return pages, metadata
