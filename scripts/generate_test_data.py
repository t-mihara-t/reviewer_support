from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def base_drawing(width: int = 1800, height: int = 1200) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (width - 80, height - 80), (0, 0, 0), 5)
    cv2.rectangle(img, (140, 140), (width - 140, height - 140), (0, 0, 0), 2)
    cv2.line(img, (260, 300), (1500, 300), (0, 0, 0), 4)
    cv2.line(img, (260, 480), (1200, 480), (0, 0, 0), 4)
    cv2.circle(img, (500, 760), 120, (0, 0, 0), 4)
    cv2.rectangle(img, (900, 650), (1300, 900), (0, 0, 0), 4)
    cv2.putText(img, "DRAWING A-001", (1070, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


def transform(img: np.ndarray, dx=0, dy=0, scale=1.0, angle=0.0) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    mat[0, 2] += dx
    mat[1, 2] += dy
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def save(path: Path, img: np.ndarray) -> None:
    Image.fromarray(img).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic drawing-diff test data.")
    parser.add_argument("output", nargs="?", default="test_data", help="Output directory")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    old = base_drawing()
    save(out / "old_base.tif", old)
    save(out / "new_identical.tif", old)

    added = old.copy(); cv2.line(added, (300, 980), (850, 980), (0, 0, 0), 4); save(out / "new_added_line.tif", added)
    deleted = old.copy(); cv2.line(deleted, (260, 480), (1200, 480), (255, 255, 255), 12); save(out / "new_deleted_line.tif", deleted)
    moved = old.copy(); cv2.line(moved, (260, 300), (1500, 300), (255, 255, 255), 12); cv2.line(moved, (260, 340), (1500, 340), (0, 0, 0), 4); save(out / "new_moved_line.tif", moved)
    save(out / "new_shifted_2mm_400dpi.tif", transform(old, dx=31, dy=31))
    save(out / "new_scaled_0_2pct.tif", transform(old, scale=1.002))
    save(out / "new_rotated_0_2deg.tif", transform(old, angle=0.2))
    multi = old.copy(); cv2.rectangle(multi, (180, 180), (1620, 1020), (0, 0, 0), 3); save(out / "old_multiple_frames.tif", multi)
    rng = np.random.default_rng(123); noisy = old.copy(); coords = rng.integers([0, 0], [old.shape[1], old.shape[0]], size=(900, 2)); noisy[coords[:, 1], coords[:, 0]] = 0; save(out / "new_noise.tif", noisy)
    Image.fromarray(old).save(out / "old_base.pdf", "PDF", resolution=400.0)
    Image.fromarray(added).save(out / "new_added_line.pdf", "PDF", resolution=400.0)
    print(f"Generated test drawings in {out}")


if __name__ == "__main__":
    main()
