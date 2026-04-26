"""Phase A tests: mask helpers."""

import numpy as np

from tracking.masks import (
    bbox_to_mask,
    extract_yolo_seg_masks,
    mask_iou,
    mask_overlap_fraction,
)


def test_mask_iou_identical_is_one():
    m = np.ones((10, 10), dtype=np.uint8)
    assert mask_iou(m, m) == 1.0


def test_mask_iou_disjoint_is_zero():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[:5, :5] = 1
    b[5:, 5:] = 1
    assert mask_iou(a, b) == 0.0


def test_mask_iou_partial_overlap():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[:6, :6] = 1
    b[4:, 4:] = 1
    # intersection 2x2=4, union = 36+36-4 = 68
    expected = 4 / 68
    assert abs(mask_iou(a, b) - expected) < 1e-6


def test_mask_iou_shape_mismatch_raises():
    a = np.ones((4, 4), dtype=np.uint8)
    b = np.ones((5, 5), dtype=np.uint8)
    try:
        mask_iou(a, b)
    except ValueError:
        return
    assert False, "expected ValueError"


def test_mask_overlap_fraction_action_in_owner():
    """An action mask fully contained in fighter A's mask -> fraction = 1."""
    fighter_a = np.zeros((20, 20), dtype=np.uint8)
    fighter_a[:, :10] = 1
    action = np.zeros((20, 20), dtype=np.uint8)
    action[5:7, 2:4] = 1
    assert mask_overlap_fraction(action, fighter_a) == 1.0


def test_mask_overlap_fraction_split():
    fighter_a = np.zeros((20, 20), dtype=np.uint8)
    fighter_a[:, :10] = 1
    fighter_b = np.zeros((20, 20), dtype=np.uint8)
    fighter_b[:, 10:] = 1
    action = np.zeros((20, 20), dtype=np.uint8)
    action[5:6, 8:12] = 1  # 2 px in A, 2 px in B
    assert abs(mask_overlap_fraction(action, fighter_a) - 0.5) < 1e-6
    assert abs(mask_overlap_fraction(action, fighter_b) - 0.5) < 1e-6


def test_bbox_to_mask():
    m = bbox_to_mask((10, 20, 30, 40), (100, 100))
    assert m.shape == (100, 100)
    assert m.sum() == 20 * 20
    assert m[25, 15] == 1
    assert m[5, 5] == 0


def test_extract_yolo_seg_masks_returns_none_for_detection_only_results():
    """Legacy YOLOv5 detection results carry no ``masks`` attribute."""

    class FakeResults:
        def __init__(self):
            self.xyxy = [None]
            self.masks = None

    r = FakeResults()
    assert extract_yolo_seg_masks(r, (480, 640)) is None


def test_extract_yolo_seg_masks_handles_v8_style_data_tensor():
    """Mimics the Ultralytics v8 results.masks.data tensor shape."""

    class FakeMasks:
        def __init__(self, arr):
            self.data = _FakeTensor(arr)

    class _FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class FakeResults:
        def __init__(self, arr):
            self.masks = FakeMasks(arr)

    arr = np.zeros((2, 32, 32), dtype=np.float32)
    arr[0, 4:10, 4:10] = 1.0
    arr[1, 20:25, 20:25] = 1.0
    results = FakeResults(arr)
    masks = extract_yolo_seg_masks(results, (64, 64))
    assert masks is not None and len(masks) == 2
    assert masks[0].shape == (64, 64)
    assert masks[0].sum() > 0
    assert masks[1].sum() > 0
