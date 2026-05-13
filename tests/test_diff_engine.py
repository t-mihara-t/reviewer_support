import pytest
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from drawing_diff_tool.app.diff_engine import DiffSettings, create_diff_image


def test_added_line_is_red_and_deleted_line_is_blue():
    old = np.zeros((80, 120), dtype=np.uint8)
    new = np.zeros_like(old)
    cv2.line(old, (10, 20), (100, 20), 255, 2)
    cv2.line(new, (10, 20), (100, 20), 255, 2)
    cv2.line(old, (10, 40), (100, 40), 255, 2)
    cv2.line(new, (10, 60), (100, 60), 255, 2)

    diff, masks = create_diff_image(old, new, DiffSettings(tolerance_px=1))

    assert np.count_nonzero(masks.common) > 0
    assert np.count_nonzero(masks.old_only) > 0
    assert np.count_nonzero(masks.new_only) > 0
    assert (diff[40, 50] == np.array([0, 0, 255])).all()
    assert (diff[60, 50] == np.array([255, 0, 0])).all()


def test_identical_drawing_has_no_colored_diff():
    old = np.zeros((80, 120), dtype=np.uint8)
    cv2.rectangle(old, (10, 10), (100, 70), 255, 2)
    diff, masks = create_diff_image(old, old.copy(), DiffSettings(tolerance_px=2))
    assert np.count_nonzero(masks.old_only) == 0
    assert np.count_nonzero(masks.new_only) == 0
    assert np.count_nonzero((diff == np.array([255, 0, 0])).all(axis=2)) == 0
    assert np.count_nonzero((diff == np.array([0, 0, 255])).all(axis=2)) == 0
