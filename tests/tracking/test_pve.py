"""Phase 2 tests: PvETracker + ImpactAttributor."""

from tracking.pve import PvETracker, ImpactAttributor


def _make_box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def test_pve_tracker_stable_id_across_frames():
    tracker = PvETracker()
    bag = (300, 100, 380, 280)
    for fi in range(10):
        person = _make_box(100 + fi * 5, 240)
        state = tracker.update(fi, [person], [bag])
        assert state["person_track"].track_id == "1"
        assert state["bag_track"].track_id == "bag"


def test_pve_tracker_detects_secondary_person_warning():
    tracker = PvETracker()
    for fi in range(11):
        a = _make_box(100, 240)
        b = _make_box(400, 240)
        tracker.update(fi, [a, b], [(500, 100, 580, 280)])
    assert any("PvP" in w for w in tracker.warnings)


def test_pve_tracker_bag_state_transitions_on_motion():
    tracker = PvETracker()
    person = _make_box(100, 240)
    # 5 resting frames.
    for fi in range(5):
        tracker.update(fi, [person], [(300, 100, 380, 280)])
    assert tracker.bag_state == "resting"
    # Now make the bag jump.
    for fi in range(5, 10):
        bag = (300 + (fi - 4) * 20, 100, 380 + (fi - 4) * 20, 280)
        tracker.update(fi, [person], [bag])
    assert tracker.bag_state in {"swinging", "struck"}


def test_pve_tracker_recovers_from_missed_person_detection():
    tracker = PvETracker()
    bag = (300, 100, 380, 280)
    for fi in range(3):
        tracker.update(fi, [_make_box(100, 240)], [bag])
    # Miss person for 2 frames.
    for fi in range(3, 5):
        state = tracker.update(fi, [], [bag])
        assert state["person_track"] is not None
    # Reappears.
    state = tracker.update(5, [_make_box(120, 240)], [bag])
    assert state["person_track"].track_id == "1"
    assert state["person_track"].age_since_seen == 0


def test_impact_attributor_landed_when_spatial_and_temporal():
    attr = ImpactAttributor()
    for fi in range(10):
        attr.record_bag_state(fi, "resting")
    attr.record_bag_state(10, "struck")
    bag = (200, 100, 320, 280)
    ev = attr.attribute(action_id=1, action_frame=10, terminal_keypoint_xy=(260, 180), bag_bbox=bag)
    assert ev.landed is True
    assert ev.confidence > 0.8


def test_impact_attributor_missed_when_no_spatial_no_temporal():
    attr = ImpactAttributor()
    for fi in range(10):
        attr.record_bag_state(fi, "resting")
    bag = (200, 100, 320, 280)
    ev = attr.attribute(action_id=2, action_frame=10, terminal_keypoint_xy=(50, 180), bag_bbox=bag)
    assert ev.landed is False


def test_impact_attributor_handles_missing_inputs():
    attr = ImpactAttributor()
    ev = attr.attribute(action_id=3, action_frame=0, terminal_keypoint_xy=None, bag_bbox=None)
    assert ev.landed is False
    assert ev.reason == "no_bag_or_keypoint"
