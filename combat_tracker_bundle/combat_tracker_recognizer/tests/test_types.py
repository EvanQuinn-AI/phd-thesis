"""Validate type invariants."""

import numpy as np
import pytest

from combat_tracker_recognizer.types import (
    KeypointFormat,
    PoseWindow,
    num_keypoints,
)


def test_pose_window_rejects_wrong_keypoint_count():
    pts = np.zeros((10, 5, 2), dtype=np.float32)
    sc = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="K="):
        PoseWindow(points=pts, scores=sc, fps=30.0, frame_start=0, frame_end=9)


def test_pose_window_rejects_score_shape_mismatch():
    pts = np.zeros((10, 13, 2), dtype=np.float32)
    sc = np.ones((10, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="scores shape"):
        PoseWindow(points=pts, scores=sc, fps=30.0, frame_start=0, frame_end=9)


def test_pose_window_num_frames_is_T():
    pts = np.zeros((20, 13, 2), dtype=np.float32)
    sc = np.ones((20, 13), dtype=np.float32)
    pw = PoseWindow(points=pts, scores=sc, fps=30.0, frame_start=0, frame_end=19)
    assert pw.num_frames == 20


def test_keypoint_count_lookup():
    assert num_keypoints(KeypointFormat.MEDIAPIPE_13) == 13
    assert num_keypoints(KeypointFormat.COCO_17) == 17
