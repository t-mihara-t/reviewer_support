import pytest
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from drawing_diff_tool.app.frame_detector import detect_frame_candidates


def test_detects_large_outer_frame():
    mask = np.zeros((600, 900), dtype=np.uint8)
    cv2.rectangle(mask, (40, 50), (860, 550), 255, 5)
    cv2.line(mask, (200, 250), (700, 250), 255, 3)
    candidates = detect_frame_candidates(mask)
    assert candidates
    frame = candidates[0]
    assert abs(frame.x - 40) < 10
    assert abs(frame.y - 50) < 10
    assert abs(frame.width - 820) < 20
    assert abs(frame.height - 500) < 20
