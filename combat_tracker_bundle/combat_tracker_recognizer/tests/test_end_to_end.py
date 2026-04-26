"""End-to-end Phase 2 test.

Synthesises a stream of pose updates and ActionAttribution events without
spinning up a real Streamlit app. Verifies:

- Empty bank → punches persist as UNKNOWN
- After seeding the bank with a jab → matching punches route KNOWN
- DB survives reconnect
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from combat_tracker_recognizer.config import RecognizerConfig, GateConfig
from combat_tracker_recognizer.integration import CombatTrackerEventConsumer
from combat_tracker_recognizer.recognizer import SubclassActionRecognizer
from combat_tracker_recognizer.tests.conftest import make_pose_window
from combat_tracker_recognizer.types import GateDecision


@dataclass
class FakeAttribution:
    """Mimics the parent's AttributedAction shape."""

    action_class: str
    owner_id: str
    target_id: Optional[str] = None


def _push_window_to_buffer(consumer: CombatTrackerEventConsumer, track_id: int,
                           start_frame: int, window) -> None:
    """Push every frame of a synthetic PoseWindow into the consumer's
    keypoint buffer for the given track."""
    landmark_names = (
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    )
    for ti in range(window.points.shape[0]):
        lm = {}
        for ki, name in enumerate(landmark_names):
            lm[name] = (
                float(window.points[ti, ki, 0]),
                float(window.points[ti, ki, 1]),
                float(window.scores[ti, ki]),
            )
        consumer.push_track_keypoints(start_frame + ti, track_id, lm)


def test_e2e_unknown_then_known(tmp_path):
    cfg = RecognizerConfig()
    cfg.store.db_path = str(tmp_path / "e2e.db")
    cfg.gate = GateConfig(known_distance_threshold=1e-4,
                          ambiguous_distance_threshold=5e-4,
                          min_margin_ratio=2.0)

    recognizer = SubclassActionRecognizer(cfg)
    consumer = CombatTrackerEventConsumer(
        recognizer=recognizer, window_before=8, window_after=4,
        session_id="sess1", fps=30.0,
    )

    # Push 12 frames of jab-pose for track 1 starting at frame 8.
    jab_window = make_pose_window("jab", T=12, seed=1)
    _push_window_to_buffer(consumer, track_id=1, start_frame=8, window=jab_window)

    # Action fires at frame 16 (mid-window).
    action = FakeAttribution(action_class="punch", owner_id="1")
    result = consumer.observe_action(action, frame_idx=16, video_ref="x.mp4")
    assert result is not None
    assert result.decision == GateDecision.UNKNOWN
    assert result.clip_id is not None

    # Seed the bank with that very pose-window's encoding.
    seed_emb = recognizer.encoder.encode(jab_window)
    recognizer.bank.add("punch", "jab", seed_emb,
                        encoder_version=recognizer.encoder.version)

    # Push another jab-like window for track 2.
    jab_window2 = make_pose_window("jab", T=12, seed=2)
    _push_window_to_buffer(consumer, track_id=2, start_frame=30, window=jab_window2)
    action2 = FakeAttribution(action_class="punch", owner_id="2")
    result2 = consumer.observe_action(action2, frame_idx=38, video_ref="x.mp4")
    assert result2.decision == GateDecision.KNOWN
    assert result2.subclass == "jab"

    # Hook should NOT route as jab.
    hook_window = make_pose_window("hook", T=12, seed=3)
    _push_window_to_buffer(consumer, track_id=3, start_frame=50, window=hook_window)
    action3 = FakeAttribution(action_class="punch", owner_id="3")
    result3 = consumer.observe_action(action3, frame_idx=58, video_ref="x.mp4")
    assert result3.decision in (GateDecision.UNKNOWN, GateDecision.AMBIGUOUS)
    assert result3.subclass != "jab"


def test_e2e_db_survives_restart(tmp_path):
    cfg = RecognizerConfig()
    cfg.store.db_path = str(tmp_path / "restart.db")

    rec = SubclassActionRecognizer(cfg)
    consumer = CombatTrackerEventConsumer(rec, session_id="sess")

    jab = make_pose_window("jab", T=12, seed=1)
    _push_window_to_buffer(consumer, track_id=1, start_frame=8, window=jab)
    consumer.observe_action(FakeAttribution(action_class="punch", owner_id="1"),
                            frame_idx=16, video_ref="x.mp4")

    # Reopen.
    rec.clipstore.close()
    rec2 = SubclassActionRecognizer(cfg)
    pending = rec2.pending_review_count(session_id="sess")
    assert pending == 1


def test_consumer_queues_action_when_history_too_short(tmp_path):
    cfg = RecognizerConfig()
    cfg.store.db_path = str(tmp_path / "q.db")

    rec = SubclassActionRecognizer(cfg)
    consumer = CombatTrackerEventConsumer(rec, window_before=8, window_after=4,
                                          session_id="s")

    # Only 3 frames of history before the action — should queue.
    jab = make_pose_window("jab", T=3, seed=1)
    _push_window_to_buffer(consumer, track_id=1, start_frame=0, window=jab)

    result = consumer.observe_action(FakeAttribution(action_class="punch", owner_id="1"),
                                     frame_idx=2, video_ref=None)
    assert result is None
    assert len(consumer._pending) == 1

    # Drive more frames.
    jab2 = make_pose_window("jab", T=10, seed=2)
    _push_window_to_buffer(consumer, track_id=1, start_frame=3, window=jab2)
    drained = consumer.tick(frame_idx=12)
    assert len(drained) == 1


def test_known_results_are_not_persisted(tmp_path):
    cfg = RecognizerConfig()
    cfg.store.db_path = str(tmp_path / "k.db")
    cfg.gate = GateConfig(known_distance_threshold=1e-4,
                          ambiguous_distance_threshold=5e-4,
                          min_margin_ratio=2.0)

    rec = SubclassActionRecognizer(cfg)
    jab_seed = rec.encoder.encode(make_pose_window("jab", seed=1))
    rec.bank.add("punch", "jab", jab_seed, encoder_version=rec.encoder.version)

    consumer = CombatTrackerEventConsumer(rec, session_id="s")
    jab2 = make_pose_window("jab", seed=2)
    _push_window_to_buffer(consumer, track_id=1, start_frame=0, window=jab2)
    result = consumer.observe_action(FakeAttribution(action_class="punch", owner_id="1"),
                                     frame_idx=8, video_ref=None)
    assert result.decision == GateDecision.KNOWN
    assert result.clip_id is None
    assert rec.pending_review_count(session_id="s") == 0
