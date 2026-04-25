"""Core data types. Kept frozen where mutation is never needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


class KeypointFormat(str, Enum):
    MEDIAPIPE_13 = "mediapipe_13"
    COCO_17 = "coco_17"


# 13-landmark order matches existing pose_analytics.py:
MEDIAPIPE_13_NAMES = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


def num_keypoints(fmt: KeypointFormat) -> int:
    return {KeypointFormat.MEDIAPIPE_13: 13, KeypointFormat.COCO_17: 17}[fmt]


@dataclass(frozen=True)
class PoseWindow:
    """Pose sequence over a temporal window.

    points: shape (T, K, 2), x/y in normalised frame coords ([0, 1]).
    scores: shape (T, K), keypoint visibility/confidence in [0, 1].
    """

    points: np.ndarray
    scores: np.ndarray
    fps: float
    frame_start: int
    frame_end: int
    keypoint_format: KeypointFormat = KeypointFormat.MEDIAPIPE_13

    def __post_init__(self) -> None:
        if self.points.ndim != 3 or self.points.shape[2] != 2:
            raise ValueError(f"points must be (T, K, 2); got {self.points.shape}")
        if self.scores.shape != self.points.shape[:2]:
            raise ValueError(
                f"scores shape {self.scores.shape} must match points (T, K)={self.points.shape[:2]}"
            )
        expected_k = num_keypoints(self.keypoint_format)
        if self.points.shape[1] != expected_k:
            raise ValueError(
                f"points K={self.points.shape[1]} != expected {expected_k} for {self.keypoint_format}"
            )
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end < frame_start")

    @property
    def num_frames(self) -> int:
        return self.points.shape[0]


@dataclass(frozen=True)
class Embedding:
    vector: np.ndarray
    encoder_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError(f"vector must be 1-D; got {self.vector.shape}")
        if self.vector.dtype != np.float32:
            object.__setattr__(self, "vector", self.vector.astype(np.float32, copy=False))


class GateDecision(str, Enum):
    KNOWN = "KNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    NOISE = "NOISE"


@dataclass
class Clip:
    parent_class: str
    pose: PoseWindow
    embedding: Embedding
    session_id: str
    id: Optional[int] = None
    video_ref: Optional[str] = None
    source_track_id: Optional[int] = None
    similarity_scores: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class SubclassResult:
    decision: GateDecision
    subclass: Optional[str]
    confidence: float
    clip_id: Optional[int]
    top_matches: list[tuple[str, float]]


@dataclass
class Cluster:
    id: int
    parent_class: str
    exemplar_clip_id: int
    member_clip_ids: list[int]
    size: int
    suggested_labels: list[tuple[str, float]] = field(default_factory=list)
    intra_distance_mean: float = 0.0


class ReviewDecision(str, Enum):
    LABEL = "LABEL"
    DISCARD = "DISCARD"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    RELABEL = "RELABEL"
