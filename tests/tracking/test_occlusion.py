"""Phase 5 tests: ClinchDetector."""

import numpy as np

from tracking.base import FeatureBank
from tracking.features import PartExtractor
from tracking.occlusion import ClinchDetector


def _box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def _frame_two_colours():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:380, 50:250] = (0, 0, 255)  # red on left
    frame[100:380, 390:590] = (255, 0, 0)  # blue on right
    return frame


def _landmarks(cx_norm):
    return {
        "nose": (cx_norm, 0.25, 0.95),
        "left_shoulder": (cx_norm - 0.05, 0.32, 0.9),
        "right_shoulder": (cx_norm + 0.05, 0.32, 0.9),
        "left_hip": (cx_norm - 0.04, 0.6, 0.9),
        "right_hip": (cx_norm + 0.04, 0.6, 0.9),
    }


def test_clinch_enters_after_high_iou_streak():
    det = ClinchDetector()
    overlap = (200, 100, 400, 300)
    overlap2 = (210, 100, 410, 300)
    for fi in range(det.cfg.clinch_min_frames):
        det.observe(fi, {"1": overlap, "2": overlap2}, num_person_detections=2)
    assert det.state.active is True


def test_clinch_enters_on_detection_collapse():
    det = ClinchDetector()
    box1 = _box(300, 200)
    box2 = _box(305, 200)
    for fi in range(det.cfg.clinch_min_frames):
        det.observe(fi, {"1": box1, "2": box2}, num_person_detections=1)
    assert det.state.active is True


def test_clinch_exits_and_starts_uncertain_window():
    det = ClinchDetector()
    overlap = (200, 100, 400, 300)
    overlap2 = (210, 100, 410, 300)
    for fi in range(det.cfg.clinch_min_frames):
        det.observe(fi, {"1": overlap, "2": overlap2}, num_person_detections=2)
    # Disengage.
    state = det.observe(100, {"1": _box(150, 240), "2": _box(490, 240)}, num_person_detections=2)
    assert state.active is False
    assert det.is_uncertain(100)
    assert det.is_uncertain(100 + det.cfg.disocclusion_uncertain_window)
    assert not det.is_uncertain(100 + det.cfg.disocclusion_uncertain_window + 1)


def test_clinch_does_not_enter_on_short_overlap():
    det = ClinchDetector()
    overlap = (200, 100, 400, 300)
    det.observe(0, {"1": overlap, "2": overlap}, num_person_detections=2)
    det.observe(1, {"1": _box(150, 240), "2": _box(490, 240)}, num_person_detections=2)
    assert det.state.active is False


def test_recover_assignment_uses_banks_not_pre_clinch_hist():
    """Bank populated with red on slot 1 / blue on slot 2 should match colours
    even when detection order is swapped."""
    det = ClinchDetector()
    frame = _frame_two_colours()
    ext = PartExtractor()
    bank_1 = FeatureBank()
    bank_2 = FeatureBank()
    bank_1.add_features(ext.extract(frame, _box(150, 240), _landmarks(150 / 640)), 0.9)
    bank_2.add_features(ext.extract(frame, _box(490, 240), _landmarks(490 / 640)), 0.9)

    # Provide detections in reversed order; recovery should still map
    # the blue detection to slot 2.
    person_dets = [_box(490, 240), _box(150, 240)]
    landmarks = [_landmarks(490 / 640), _landmarks(150 / 640)]
    out = det.recover_assignment(frame, person_dets, landmarks, {"1": bank_1, "2": bank_2})
    assert out["1"] == 1  # red detection (second in list)
    assert out["2"] == 0  # blue detection (first in list)
    assert det.state.last_recovery_confidence >= 0.0
