"""Tests for PersonFilter."""

from tracking.person_filter import PersonFilter


def _box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def test_drops_small_background_person_by_area_when_three_dets():
    pf = PersonFilter(min_area_ratio=0.3)
    fighter_a = _box(150, 240, w=200, h=500)
    fighter_b = _box(490, 240, w=180, h=480)
    background = _box(640, 300, w=40, h=80)  # tiny ratio
    kept, _, decisions = pf.filter([fighter_a, fighter_b, background])
    assert background not in kept
    assert {d.reason for d in decisions} >= {"kept", "small_area"}


def test_two_detections_always_pass_through():
    """Filters do not fire with <=2 detections — fighters in pauses must not be dropped."""
    pf = PersonFilter(min_area_ratio=0.3)
    fighter = _box(150, 240, w=200, h=500)
    background_looking = _box(500, 300, w=40, h=80)  # would be dropped at 3+ dets
    kept, _, decisions = pf.filter([fighter, background_looking])
    assert kept == [fighter, background_looking]
    assert all(d.reason == "kept" for d in decisions)


def test_keeps_both_when_sizes_similar():
    pf = PersonFilter(min_area_ratio=0.3)
    a = _box(150, 240, w=200, h=500)
    b = _box(490, 240, w=180, h=480)
    kept, _, _ = pf.filter([a, b])
    assert set(kept) == {a, b}


def test_drops_stationary_tracklet_after_min_observations_when_three_dets():
    pf = PersonFilter(stationary_variance_thresh=10.0, stationary_min_observations=5)
    fighter_static = _box(490, 240)
    last_decisions = None
    for fi in range(10):
        fighter_a = _box(150 + fi * 8, 240)
        fighter_b = _box(300 + fi * 4, 240)
        kept, _, last_decisions = pf.filter([fighter_a, fighter_b, fighter_static])
    reasons = {d.reason for d in last_decisions}
    assert "stationary" in reasons


def test_does_not_drop_moving_fighter_as_stationary():
    pf = PersonFilter(stationary_variance_thresh=10.0, stationary_min_observations=5)
    last_decisions = None
    for fi in range(10):
        fighter_a = _box(150 + fi * 8, 240)
        fighter_b = _box(490 - fi * 6, 240)
        fighter_c = _box(320 + fi * 5, 240)
        kept, _, last_decisions = pf.filter([fighter_a, fighter_b, fighter_c])
    # All three are moving — none should be flagged stationary.
    assert all(d.reason in {"kept", "small_area"} for d in last_decisions)
    assert not any(d.reason == "stationary" for d in last_decisions)


def test_landmarks_aligned_with_kept_detections_three_dets():
    pf = PersonFilter(min_area_ratio=0.3)
    big_a = _box(150, 240, w=200, h=500)
    big_b = _box(450, 240, w=200, h=500)
    small = _box(640, 300, w=40, h=80)
    kept, kept_lm, _ = pf.filter([big_a, big_b, small],
                                 landmarks_per_person=[{"a": 1}, {"b": 2}, {"c": 3}])
    assert small not in kept
    assert {"c": 3} not in kept_lm


def test_empty_input_is_safe():
    pf = PersonFilter()
    kept, kept_lm, decisions = pf.filter([])
    assert kept == [] and kept_lm == [] and decisions == []
