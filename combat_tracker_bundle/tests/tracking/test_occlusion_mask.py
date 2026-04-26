"""Phase C tests: ClinchDetector with mask-IoU."""

import numpy as np

from tracking.occlusion import ClinchDetector


def _box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def _disjoint_masks(shape=(480, 640)):
    """Two non-overlapping masks at the same y but separated in x."""
    a = np.zeros(shape, dtype=np.uint8)
    a[100:380, 50:250] = 1
    b = np.zeros(shape, dtype=np.uint8)
    b[100:380, 390:590] = 1
    return a, b


def _overlapping_masks(shape=(480, 640), overlap_x=80):
    """Two masks that share ``overlap_x`` columns near the centre."""
    a = np.zeros(shape, dtype=np.uint8)
    a[100:380, 200:320 + overlap_x // 2] = 1
    b = np.zeros(shape, dtype=np.uint8)
    b[100:380, 320 - overlap_x // 2:440] = 1
    return a, b


def test_bbox_overlap_does_not_trigger_when_masks_separate():
    """Two bboxes can overlap heavily while their pixel masks remain
    cleanly separated (e.g. fighters close to camera but not touching).
    Mask-aware path must not enter clinch."""
    det = ClinchDetector()
    big_box = (200, 100, 440, 380)  # one bbox covering both fighters
    a, b = _disjoint_masks()
    for fi in range(det.cfg.clinch_min_frames * 2):
        det.observe(
            fi,
            {"1": big_box, "2": big_box},
            num_person_detections=2,
            slot_masks={"1": a, "2": b},
        )
    assert det.state.active is False


def test_mask_overlap_does_trigger_clinch():
    det = ClinchDetector()
    big_box = (200, 100, 440, 380)
    a, b = _overlapping_masks(overlap_x=160)  # heavy mask overlap
    for fi in range(det.cfg.clinch_min_frames + 2):
        det.observe(
            fi,
            {"1": big_box, "2": big_box},
            num_person_detections=2,
            slot_masks={"1": a, "2": b},
        )
    assert det.state.active is True


def test_mask_path_falls_back_to_bbox_when_masks_missing():
    """Backwards compatibility: omitting slot_masks restores bbox-IoU behaviour."""
    det = ClinchDetector()
    overlap = (200, 100, 400, 300)
    overlap2 = (210, 100, 410, 300)
    for fi in range(det.cfg.clinch_min_frames):
        det.observe(fi, {"1": overlap, "2": overlap2}, num_person_detections=2)
    assert det.state.active is True


def test_mask_det_separation_drives_exit():
    """After clinching via masks, a frame with two clearly-separate
    detection masks must force-exit even if predicted slot masks still overlap."""
    det = ClinchDetector()
    big_box = (200, 100, 440, 380)
    overlap_a, overlap_b = _overlapping_masks(overlap_x=180)
    for fi in range(det.cfg.clinch_min_frames + 2):
        det.observe(
            fi,
            {"1": big_box, "2": big_box},
            num_person_detections=2,
            slot_masks={"1": overlap_a, "2": overlap_b},
        )
    assert det.state.active is True

    # Detection masks now clearly apart.
    sep_a, sep_b = _disjoint_masks()
    state = det.observe(
        100,
        {"1": big_box, "2": big_box},  # predicted slot masks still merged
        num_person_detections=2,
        person_dets=[_box(150, 240), _box(490, 240)],
        slot_masks={"1": overlap_a, "2": overlap_b},
        person_masks=[sep_a, sep_b],
    )
    assert state.active is False
    assert det.is_uncertain(100)
