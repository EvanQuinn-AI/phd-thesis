"""Phase 6 tests: ActionOwnership."""

from tracking.occlusion import ClinchDetector
from tracking.ownership import ActionOwnership


def _box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def test_punch_owner_inferred_from_wrist_proximity():
    own = ActionOwnership()
    frame_size = (640, 480)
    tracks = {"1": {"box": _box(150, 240)}, "2": {"box": _box(490, 240)}}
    landmarks = {
        "1": {"right_wrist": (0.3, 0.45, 0.9), "left_wrist": (0.18, 0.55, 0.9)},
        "2": {"right_wrist": (0.7, 0.55, 0.9), "left_wrist": (0.78, 0.45, 0.9)},
    }
    # Action centre near track 1's right wrist (x=192, y=216).
    action_box = (180, 200, 220, 240)
    out = own.assign(action_id=1, action_class="punch", action_box=action_box,
                     frame_idx=10, frame_size=frame_size,
                     tracks=tracks, landmarks_per_track=landmarks)
    assert out.owner_id == "1"
    assert out.method == "kinematic"
    assert out.target_id == "2"


def test_kick_owner_inferred_from_ankle_proximity():
    own = ActionOwnership()
    frame_size = (640, 480)
    tracks = {"1": {"box": _box(150, 240)}, "2": {"box": _box(490, 240)}}
    landmarks = {
        "1": {"left_ankle": (0.25, 0.85, 0.9), "right_ankle": (0.20, 0.85, 0.9)},
        "2": {"left_ankle": (0.75, 0.85, 0.9), "right_ankle": (0.80, 0.85, 0.9)},
    }
    action_box = (300, 380, 340, 420)  # closer to track 1's right ankle than track 2's
    out = own.assign(action_id=2, action_class="kick", action_box=action_box,
                     frame_idx=10, frame_size=frame_size,
                     tracks=tracks, landmarks_per_track=landmarks)
    assert out.owner_id == "1"


def test_falls_back_to_centroid_when_no_landmarks():
    own = ActionOwnership()
    frame_size = (640, 480)
    tracks = {"1": {"box": _box(150, 240)}, "2": {"box": _box(490, 240)}}
    action_box = (130, 220, 170, 260)  # centre 150,240 -> inside track 1
    out = own.assign(action_id=3, action_class="punch", action_box=action_box,
                     frame_idx=10, frame_size=frame_size,
                     tracks=tracks, landmarks_per_track={"1": None, "2": None})
    assert out.owner_id == "1"
    assert out.method == "centroid_fallback"


def test_unattributed_when_centroid_in_two_boxes():
    own = ActionOwnership()
    frame_size = (640, 480)
    tracks = {"1": {"box": (100, 100, 400, 400)}, "2": {"box": (300, 100, 600, 400)}}
    action_box = (340, 240, 360, 260)  # centre 350,250 -> inside both
    out = own.assign(action_id=4, action_class="punch", action_box=action_box,
                     frame_idx=10, frame_size=frame_size,
                     tracks=tracks, landmarks_per_track={"1": None, "2": None})
    assert out.owner_id is None
    assert out.reason == "ambiguous_centroid"
    assert out.method == "unattributed"


def test_clinch_suppresses_attribution():
    clinch = ClinchDetector()
    # Force into clinch.
    overlap = (200, 100, 400, 300)
    for fi in range(clinch.cfg.clinch_min_frames):
        clinch.observe(fi, {"1": overlap, "2": overlap}, num_person_detections=1)
    own = ActionOwnership(clinch_detector=clinch)
    out = own.assign(action_id=5, action_class="punch", action_box=(280, 180, 320, 220),
                     frame_idx=20, frame_size=(640, 480),
                     tracks={"1": {"box": overlap}, "2": {"box": overlap}},
                     landmarks_per_track={})
    assert out.owner_id is None
    assert out.reason == "clinch"


def test_target_resolves_to_bag_when_only_bag_present():
    own = ActionOwnership()
    tracks = {"1": {"box": _box(150, 240)}}
    landmarks = {"1": {"right_wrist": (0.30, 0.45, 0.9)}}
    bag = _box(490, 240)
    out = own.assign(action_id=6, action_class="punch", action_box=(180, 200, 220, 240),
                     frame_idx=10, frame_size=(640, 480),
                     tracks=tracks, landmarks_per_track=landmarks, bag_box=bag)
    assert out.owner_id == "1"
    assert out.target_id == "bag"
