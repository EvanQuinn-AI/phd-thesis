"""Shared fixtures for tracking tests."""

import numpy as np
import pytest


@pytest.fixture
def two_rect_frame():
    """Frame with two solid colour rectangles (red on left, blue on right)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:380, 50:250] = (0, 0, 255)
    frame[100:380, 390:590] = (255, 0, 0)
    return frame


@pytest.fixture
def two_rect_bboxes():
    return [(50, 100, 250, 380), (390, 100, 590, 380)]


@pytest.fixture
def fake_landmarks_two_people():
    """MediaPipe-style normalised landmark dicts for two people in two_rect_frame."""

    def make(cx_norm):
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

    return [make(150 / 640), make(490 / 640)]
