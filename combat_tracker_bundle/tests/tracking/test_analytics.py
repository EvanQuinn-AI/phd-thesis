"""Tests for FighterAnalytics."""

import numpy as np

from tracking.analytics import FighterAnalytics


def _box(cx, cy, w=80, h=200):
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def test_throws_and_landed_aggregate_per_fighter():
    a = FighterAnalytics(fps=30.0, frame_size=(640, 480))
    a.record_action_thrown("1", "cross")
    a.record_action_thrown("1", "hook")
    a.record_action_landed("1", "2", "cross")
    a.record_action_thrown("2", "kick")
    s = a.summary()
    assert s["fighters"]["1"]["throws_total"] == 2
    assert s["fighters"]["1"]["throws_by_class"] == {"cross": 1, "hook": 1}
    assert s["fighters"]["1"]["landed_total"] == 1
    assert s["fighters"]["1"]["hit_rate"] == 0.5
    assert s["fighters"]["2"]["shots_received_total"] == 1
    assert s["fighters"]["2"]["throws_total"] == 1


def test_centroid_travel_and_speed():
    a = FighterAnalytics(fps=30.0, frame_size=(640, 480))
    # Walk fighter 1 ten px to the right per frame.
    for fi in range(10):
        a.observe_frame(
            slot_bboxes={"1": _box(100 + 10 * fi, 240), "2": _box(490, 240)},
            slot_landmarks={"1": None, "2": None},
            clinch_active=False, action_classes_present=[],
        )
    s = a.summary()
    f1 = s["fighters"]["1"]
    # 9 transitions × 10 px each = 90 px.
    assert abs(f1["travel_distance_px"] - 90) < 1
    # 10 px per (1/30 s) = 300 px/sec.
    assert abs(f1["mean_centroid_speed_pps"] - 300.0) < 1.0


def test_clinch_frame_counter_per_fighter():
    a = FighterAnalytics(fps=30.0, frame_size=(640, 480))
    for fi in range(20):
        a.observe_frame(
            slot_bboxes={"1": _box(100, 240), "2": _box(490, 240)},
            slot_landmarks={"1": None, "2": None},
            clinch_active=(fi >= 10),
            action_classes_present=[],
        )
    s = a.summary()
    assert s["fighters"]["1"]["time_in_clinch_frames"] == 10
    assert s["fighters"]["2"]["time_in_clinch_frames"] == 10
    assert abs(s["fighters"]["1"]["time_in_clinch_seconds"] - 10 / 30.0) < 1e-6


def test_engagement_distance_and_strike_range_counter():
    a = FighterAnalytics(fps=30.0, frame_size=(640, 480))
    # Frames close together.
    for _ in range(5):
        a.observe_frame(
            slot_bboxes={"1": _box(280, 240), "2": _box(360, 240)},
            slot_landmarks={"1": None, "2": None},
            clinch_active=False, action_classes_present=[],
        )
    # Frames far apart.
    for _ in range(5):
        a.observe_frame(
            slot_bboxes={"1": _box(50, 240), "2": _box(590, 240)},
            slot_landmarks={"1": None, "2": None},
            clinch_active=False, action_classes_present=[],
        )
    s = a.summary()
    eng = s["engagement"]
    assert eng["frames_within_strike_range"] == 5
    assert eng["mean_distance_between_fighters_px"] > 80


def test_wrist_speed_from_landmarks():
    a = FighterAnalytics(fps=30.0, frame_size=(640, 480))
    # Wrist moves 32 px (in normalised coords: 0.05 * 640) per frame.
    for fi in range(10):
        lm = {
            "left_wrist": (0.1 + 0.05 * fi, 0.5, 0.9),
            "right_wrist": (0.9 - 0.05 * fi, 0.5, 0.9),
        }
        a.observe_frame(
            slot_bboxes={"1": _box(150, 240), "2": _box(490, 240)},
            slot_landmarks={"1": lm, "2": None},
            clinch_active=False, action_classes_present=[],
        )
    s = a.summary()
    # 32 px / (1/30 s) = 960 px/s, both wrists.
    assert abs(s["fighters"]["1"]["mean_wrist_speed_pps"] - 960.0) < 5.0


def test_guards_active_counts_frames_with_guard_class():
    a = FighterAnalytics(fps=30.0, frame_size=(640, 480))
    for fi in range(10):
        present = ["high-guard"] if fi % 2 == 0 else []
        a.observe_frame(
            slot_bboxes={"1": _box(100, 240), "2": _box(490, 240)},
            slot_landmarks={"1": None, "2": None},
            clinch_active=False, action_classes_present=present,
        )
    s = a.summary()
    assert s["fighters"]["1"]["guards_active_frames"] == 5
