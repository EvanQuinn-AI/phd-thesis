"""Shared fixtures: synthetic pose-window generators."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from combat_tracker_recognizer.types import (
    Embedding,
    KeypointFormat,
    PoseWindow,
)


def _baseline_pose() -> np.ndarray:
    """A neutral standing pose, normalised coords (0..1). 13 keypoints."""
    return np.array([
        [0.50, 0.10],  # nose
        [0.45, 0.18],  # l_shoulder
        [0.55, 0.18],  # r_shoulder
        [0.42, 0.30],  # l_elbow
        [0.58, 0.30],  # r_elbow
        [0.40, 0.42],  # l_wrist
        [0.60, 0.42],  # r_wrist
        [0.46, 0.50],  # l_hip
        [0.54, 0.50],  # r_hip
        [0.45, 0.70],  # l_knee
        [0.55, 0.70],  # r_knee
        [0.45, 0.90],  # l_ankle
        [0.55, 0.90],  # r_ankle
    ], dtype=np.float32)


def make_pose_window(
    name: str,
    T: int = 12,
    fps: float = 30.0,
    seed: int = 0,
) -> PoseWindow:
    """Synthesise a pose window matching one of:
       'jab'   — left wrist extends straight forward (right side of frame)
       'hook'  — left wrist arcs across body
       'cross' — right wrist extends straight forward
       'idle'  — neutral pose with mild noise
       'kick_r' — right ankle lifts and swings
    """
    rng = np.random.default_rng(seed)
    points = np.tile(_baseline_pose(), (T, 1, 1)).astype(np.float32)
    scores = np.full((T, 13), 0.95, dtype=np.float32)

    t = np.linspace(0, 1, T)
    if name == "jab":
        # Left wrist (idx 5) shoots from x=0.40 to x=0.20 over the window.
        points[:, 5, 0] = 0.40 - 0.20 * np.sin(t * np.pi)
        points[:, 5, 1] = 0.42 - 0.05 * np.sin(t * np.pi)
        # Left elbow trails.
        points[:, 3, 0] = 0.42 - 0.10 * np.sin(t * np.pi)
    elif name == "hook":
        # Left wrist arcs from 0.40 to 0.55, lifting then dropping.
        points[:, 5, 0] = 0.40 + 0.15 * np.sin(t * np.pi)
        points[:, 5, 1] = 0.42 - 0.10 * np.sin(t * np.pi) ** 2
        points[:, 3, 0] = 0.42 + 0.05 * np.sin(t * np.pi)
    elif name == "cross":
        points[:, 6, 0] = 0.60 + 0.20 * np.sin(t * np.pi)
        points[:, 6, 1] = 0.42 - 0.05 * np.sin(t * np.pi)
        points[:, 4, 0] = 0.58 + 0.10 * np.sin(t * np.pi)
    elif name == "kick_r":
        points[:, 12, 0] = 0.55 + 0.20 * np.sin(t * np.pi)
        points[:, 12, 1] = 0.90 - 0.40 * np.sin(t * np.pi)
        points[:, 10, 1] = 0.70 - 0.20 * np.sin(t * np.pi)
    elif name == "idle":
        pass
    else:
        raise ValueError(f"unknown synthetic action {name!r}")

    points += rng.normal(0, 0.005, size=points.shape).astype(np.float32)
    return PoseWindow(
        points=points, scores=scores, fps=fps,
        frame_start=0, frame_end=T - 1,
        keypoint_format=KeypointFormat.MEDIAPIPE_13,
    )


@pytest.fixture
def jab_window():
    return make_pose_window("jab", seed=1)


@pytest.fixture
def hook_window():
    return make_pose_window("hook", seed=2)


@pytest.fixture
def cross_window():
    return make_pose_window("cross", seed=3)


@pytest.fixture
def kick_window():
    return make_pose_window("kick_r", seed=4)


@pytest.fixture
def make_window():
    return make_pose_window


@pytest.fixture
def fake_embedding():
    def _make(d: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(d).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        return Embedding(vector=v, encoder_version="test", created_at=datetime.utcnow())
    return _make
