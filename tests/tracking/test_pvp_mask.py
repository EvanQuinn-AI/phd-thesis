"""Phase D tests: PvPTracker thread masks through."""

import numpy as np

from tracking.pvp import PvPTracker


def _box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def _frame_two_strips():
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[:, :320] = (0, 0, 255)   # red on left
    f[:, 320:] = (255, 0, 0)   # blue on right
    return f


def _landmarks(cx_norm: float):
    return {
        "nose": (cx_norm, 0.25, 0.95),
        "left_shoulder": (cx_norm - 0.05, 0.32, 0.9),
        "right_shoulder": (cx_norm + 0.05, 0.32, 0.9),
        "left_elbow": (cx_norm - 0.07, 0.45, 0.8),
        "right_elbow": (cx_norm + 0.07, 0.45, 0.8),
        "left_wrist": (cx_norm - 0.08, 0.55, 0.8),
        "right_wrist": (cx_norm + 0.08, 0.55, 0.8),
        "left_hip": (cx_norm - 0.04, 0.6, 0.9),
        "right_hip": (cx_norm + 0.04, 0.6, 0.9),
        "left_knee": (cx_norm - 0.04, 0.72, 0.8),
        "right_knee": (cx_norm + 0.04, 0.72, 0.8),
        "left_ankle": (cx_norm - 0.04, 0.85, 0.7),
        "right_ankle": (cx_norm + 0.04, 0.85, 0.7),
    }


def _mask_for(bbox: tuple, frame_shape=(480, 640)):
    h, w = frame_shape
    m = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    m[y1:y2, x1:x2] = 1
    return m


def test_pvp_tracker_runs_with_masks():
    tracker = PvPTracker()
    boxes = [_box(150, 240), _box(490, 240)]
    masks = [_mask_for(b) for b in boxes]
    landmarks = [_landmarks(150 / 640), _landmarks(490 / 640)]
    for _ in range(tracker.cfg.anchor_window_frames + 5):
        tracker.update(_frame_two_strips(), boxes, landmarks, masks_per_person=masks)
    out = tracker.as_legacy_tracked_dict()
    assert out["1"]["box"] is not None and out["2"]["box"] is not None
    assert tracker.slots["1"].last_mask is not None
    assert tracker.slots["2"].last_mask is not None


def test_pvp_tracker_mask_argument_is_optional():
    """Existing call signature with no masks must still work."""
    tracker = PvPTracker()
    boxes = [_box(150, 240), _box(490, 240)]
    landmarks = [_landmarks(150 / 640), _landmarks(490 / 640)]
    for _ in range(tracker.cfg.anchor_window_frames + 5):
        tracker.update(_frame_two_strips(), boxes, landmarks)
    out = tracker.as_legacy_tracked_dict()
    assert out["1"]["box"] is not None and out["2"]["box"] is not None


def test_mask_path_keeps_clean_feature_bank():
    """With overlapping bboxes but disjoint masks, the FeatureBank should
    still hold red histograms for slot 1 and blue for slot 2."""
    tracker = PvPTracker()
    boxes = [(100, 100, 360, 380), (280, 100, 540, 380)]  # bboxes overlap
    masks = [_mask_for((100, 100, 320, 380)), _mask_for((320, 100, 540, 380))]
    landmarks = [_landmarks(0.30), _landmarks(0.60)]
    for _ in range(tracker.cfg.anchor_window_frames + 8):
        tracker.update(_frame_two_strips(), boxes, landmarks, masks_per_person=masks)
    bank_1 = tracker.slots["1"].feature_bank
    bank_2 = tracker.slots["2"].feature_bank
    # Either bank has at least one region with samples.
    assert bank_1.has_region("torso") or bank_1.has_region("trunks")
    assert bank_2.has_region("torso") or bank_2.has_region("trunks")
    # Slot 1 should match a fresh red sample better than slot 2 does.
    from tracking.features import PartExtractor
    ext = PartExtractor()
    fresh_red = ext.extract(_frame_two_strips(), _box(150, 240),
                            _landmarks(150 / 640),
                            mask=_mask_for(_box(150, 240)))
    s1 = bank_1.score(fresh_red)
    s2 = bank_2.score(fresh_red)
    assert s1 < s2, f"slot 1 (red bank) should match a red sample: {s1} vs {s2}"
