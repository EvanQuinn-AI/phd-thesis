"""Phase 7 tests: integration adapter."""

import os

import numpy as np

from tracking.integration import is_v2_enabled, update_two_person_ids_v2
from tracking.pvp import PvPTracker


def _box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def _frame():
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[100:380, 50:250] = (0, 0, 255)
    f[100:380, 390:590] = (255, 0, 0)
    return f


def _legacy_tracked():
    return {
        "1": {"box": None, "hist": None, "action_counts": {"cross": 0, "hook": 0},
              "last_hit_frame": {"cross": -1, "hook": -1}},
        "2": {"box": None, "hist": None, "action_counts": {"cross": 0, "hook": 0},
              "last_hit_frame": {"cross": -1, "hook": -1}},
    }


def test_v2_enabled_env_var(monkeypatch):
    monkeypatch.delenv("USE_TRACKING_V2", raising=False)
    assert is_v2_enabled() is False
    monkeypatch.setenv("USE_TRACKING_V2", "1")
    assert is_v2_enabled() is True
    monkeypatch.setenv("USE_TRACKING_V2", "0")
    assert is_v2_enabled() is False


def test_adapter_preserves_action_counts_and_updates_box():
    tracked = _legacy_tracked()
    tracked["1"]["action_counts"]["cross"] = 5
    tracked["2"]["last_hit_frame"]["hook"] = 42
    tracker = PvPTracker()
    boxes = [_box(150, 240), _box(490, 240)]
    # Run through the anchoring window plus one steady-state frame.
    for _ in range(tracker.cfg.anchor_window_frames + 1):
        update_two_person_ids_v2(_frame(), boxes, tracked, tracker, landmarks_per_person=None)

    assert tracked["1"]["action_counts"]["cross"] == 5  # preserved
    assert tracked["2"]["last_hit_frame"]["hook"] == 42  # preserved
    # boxes populated by the new tracker.
    assert tracked["1"]["box"] is not None
    assert tracked["2"]["box"] is not None


def test_adapter_works_without_landmarks():
    tracked = _legacy_tracked()
    tracker = PvPTracker()
    for _ in range(tracker.cfg.anchor_window_frames + 5):
        update_two_person_ids_v2(_frame(), [_box(150, 240), _box(490, 240)], tracked, tracker)
    assert tracked["1"]["box"] is not None
    assert tracked["2"]["box"] is not None
