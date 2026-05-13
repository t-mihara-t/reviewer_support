import pytest
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from drawing_diff_tool.app.aligner import align_image_to_frame
from drawing_diff_tool.app.frame_detector import manual_candidate


def test_aligns_shifted_frame_center():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (70, 60), (350, 260), (0, 0, 0), 3)
    source = manual_candidate([(70, 60), (350, 60), (350, 260), (70, 260)])
    target = manual_candidate([(50, 40), (330, 40), (330, 240), (50, 240)])
    warped, params, _ = align_image_to_frame(img, source, target, 400, img.shape)
    assert warped.shape == img.shape
    assert abs(params.translation_x_px + 20) < 1
    assert abs(params.translation_y_px + 20) < 1
