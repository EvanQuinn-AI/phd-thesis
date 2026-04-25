"""Phase 4 tests: PvPTracker."""

import numpy as np

from tracking.pvp import PvPTracker


def _make_box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def _frame_with_two_colours(shift_a=0, shift_b=0):
    """Two solid rectangles, red on left and blue on right; positions can shift."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:380, 50 + shift_a:250 + shift_a] = (0, 0, 255)
    frame[100:380, 390 + shift_b:590 + shift_b] = (255, 0, 0)
    return frame


def _landmarks(cx_norm):
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


def test_pvp_tracker_anchors_after_window():
    tracker = PvPTracker()
    boxes = [_make_box(150, 240), _make_box(490, 240)]
    landmarks = [_landmarks(150 / 640), _landmarks(490 / 640)]
    for _ in range(tracker.cfg.anchor_window_frames):
        tracker.update(_frame_with_two_colours(), boxes, landmarks)
    assert tracker.anchored
    out = tracker.as_legacy_tracked_dict()
    assert out["1"]["box"] is not None
    assert out["2"]["box"] is not None
    assert out["1"]["start_region"] == "left_half"
    assert out["2"]["start_region"] == "right_half"


def test_pvp_tracker_id_stable_under_drift():
    tracker = PvPTracker()
    landmarks_a = _landmarks(150 / 640)
    landmarks_b = _landmarks(490 / 640)
    for fi in range(40):
        boxes = [_make_box(150 + fi, 240), _make_box(490 - fi, 240)]
        tracker.update(_frame_with_two_colours(shift_a=fi, shift_b=-fi), boxes, [landmarks_a, landmarks_b])
    out = tracker.as_legacy_tracked_dict()
    # Slot 1 (left start) should still be the leftmost box.
    assert out["1"]["box"][0] < out["2"]["box"][0]


def test_pvp_tracker_iou_match_survives_swap_in_detection_order():
    tracker = PvPTracker()
    landmarks_a = _landmarks(150 / 640)
    landmarks_b = _landmarks(490 / 640)
    boxes = [_make_box(150, 240), _make_box(490, 240)]
    for _ in range(tracker.cfg.anchor_window_frames):
        tracker.update(_frame_with_two_colours(), boxes, [landmarks_a, landmarks_b])
    # Swap detection order.
    swapped = [boxes[1], boxes[0]]
    tracker.update(_frame_with_two_colours(), swapped, [landmarks_b, landmarks_a])
    out = tracker.as_legacy_tracked_dict()
    assert out["1"]["box"][0] < out["2"]["box"][0]


def test_pvp_tracker_holds_track_through_missed_frame():
    tracker = PvPTracker()
    landmarks_a = _landmarks(150 / 640)
    landmarks_b = _landmarks(490 / 640)
    boxes = [_make_box(150, 240), _make_box(490, 240)]
    for _ in range(tracker.cfg.anchor_window_frames):
        tracker.update(_frame_with_two_colours(), boxes, [landmarks_a, landmarks_b])
    # Drop slot 2's detection for one frame.
    tracker.update(_frame_with_two_colours(), [boxes[0]], [landmarks_a])
    out = tracker.as_legacy_tracked_dict()
    assert out["2"]["box"] is not None
    assert tracker.slots["2"].age_since_seen == 1


def test_pvp_tracker_drops_slot_after_max_age():
    tracker = PvPTracker()
    landmarks_a = _landmarks(150 / 640)
    landmarks_b = _landmarks(490 / 640)
    boxes = [_make_box(150, 240), _make_box(490, 240)]
    for _ in range(tracker.cfg.anchor_window_frames):
        tracker.update(_frame_with_two_colours(), boxes, [landmarks_a, landmarks_b])
    for _ in range(tracker.cfg.max_age_frames + 5):
        tracker.update(_frame_with_two_colours(), [boxes[0]], [landmarks_a])
    out = tracker.as_legacy_tracked_dict()
    assert out["2"]["box"] is None


def test_pvp_tracker_legacy_dict_shape_compatible():
    tracker = PvPTracker()
    landmarks_a = _landmarks(150 / 640)
    landmarks_b = _landmarks(490 / 640)
    boxes = [_make_box(150, 240), _make_box(490, 240)]
    tracker.update(_frame_with_two_colours(), boxes, [landmarks_a, landmarks_b])
    out = tracker.as_legacy_tracked_dict()
    assert set(out.keys()) == {"1", "2"}
    for tid in ("1", "2"):
        assert "box" in out[tid]
        assert "hist" in out[tid]
        assert "action_counts" in out[tid]


def test_pvp_tracker_no_landmarks_falls_back_gracefully():
    tracker = PvPTracker()
    boxes = [_make_box(150, 240), _make_box(490, 240)]
    for _ in range(tracker.cfg.anchor_window_frames + 5):
        tracker.update(_frame_with_two_colours(), boxes, None)
    out = tracker.as_legacy_tracked_dict()
    assert out["1"]["box"] is not None and out["2"]["box"] is not None
