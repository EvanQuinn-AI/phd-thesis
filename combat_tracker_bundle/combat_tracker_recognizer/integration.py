"""CombatTrackerEventConsumer: bridge between the parent tracker and the recognizer.

Maintains a per-track keypoint ring buffer. When the parent emits an
``AttributedAction`` event, builds a ``PoseWindow`` from the buffer
around the action frame and calls ``recognizer.observe(...)``.

The parent package is named ``tracking/`` in this repo (see DEVIATIONS.md
D1). This consumer imports from ``tracking`` rather than the
plan's ``combat_tracker``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from combat_tracker_recognizer.recognizer import SubclassActionRecognizer
from combat_tracker_recognizer.types import (
    KeypointFormat,
    PoseWindow,
    SubclassResult,
)


_LANDMARK_ORDER = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass
class _PendingAction:
    parent_class: str
    track_id: int
    fired_at_frame: int
    target_frame: int  # frame at which we expect window_after to be filled
    video_ref: Optional[str]


class CombatTrackerEventConsumer:
    """Wraps the parent tracker. Public surface: ``update`` and ``flush``."""

    def __init__(
        self,
        recognizer: SubclassActionRecognizer,
        window_before: int = 8,
        window_after: int = 4,
        session_id: str = "default",
        fps: float = 30.0,
    ):
        self.recognizer = recognizer
        self.window_before = window_before
        self.window_after = window_after
        self.session_id = session_id
        self.fps = fps
        # buffers[track_id] = deque[(frame_idx, landmarks_dict)]
        self._buffers: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window_before + window_after + 8)
        )
        self._pending: list[_PendingAction] = []
        self._last_seen_frame: int = -1

    def push_track_keypoints(
        self,
        frame_idx: int,
        track_id: int,
        landmarks: Optional[dict],
    ) -> None:
        """Per-frame update. ``landmarks`` is ``{name: (x_norm, y_norm, vis)}`` or None."""
        if landmarks is None:
            return
        self._buffers[track_id].append((frame_idx, landmarks))
        self._last_seen_frame = frame_idx

    def observe_action(
        self,
        attribution,
        frame_idx: int,
        video_ref: Optional[str] = None,
    ) -> Optional[SubclassResult]:
        """Called once per ``AttributedAction`` event from the parent tracker.

        Either dispatches to the recognizer immediately (sufficient history)
        or queues the event until ``window_after`` further frames arrive.
        """
        track_id = self._normalise_track_id(attribution.owner_id)
        if track_id is None:
            return None
        buf = self._buffers.get(track_id)
        if not buf or len(buf) < self.window_before:
            self._pending.append(_PendingAction(
                parent_class=attribution.action_class,
                track_id=track_id,
                fired_at_frame=frame_idx,
                target_frame=frame_idx + self.window_after,
                video_ref=video_ref,
            ))
            return None
        return self._dispatch(attribution.action_class, track_id,
                              frame_idx, video_ref)

    def tick(self, frame_idx: int, video_ref: Optional[str] = None) -> list[SubclassResult]:
        """Called once per frame after ``observe_action`` and any
        ``push_track_keypoints``. Drains pending actions whose
        ``window_after`` frames have arrived. Drained actions dispatch
        with whatever buffer history we have — even a short pre-window
        is better than dropping the action (which is what would happen
        if the action fires near the start of the video)."""
        results: list[SubclassResult] = []
        still_pending: list[_PendingAction] = []
        for p in self._pending:
            if frame_idx >= p.target_frame:
                r = self._dispatch(p.parent_class, p.track_id, p.fired_at_frame,
                                   p.video_ref, force=True)
                if r is not None:
                    results.append(r)
            else:
                still_pending.append(p)
        self._pending = still_pending
        return results

    def flush(self) -> list[SubclassResult]:
        """End-of-video: process any pending events with whatever buffer we have."""
        results: list[SubclassResult] = []
        for p in self._pending:
            r = self._dispatch(p.parent_class, p.track_id, p.fired_at_frame,
                               p.video_ref, force=True)
            if r is not None:
                results.append(r)
        self._pending.clear()
        return results

    # ---- Internal ------------------------------------------------------

    @staticmethod
    def _normalise_track_id(owner_id) -> Optional[int]:
        if owner_id is None:
            return None
        try:
            return int(owner_id)
        except (TypeError, ValueError):
            return None

    def _dispatch(
        self,
        parent_class: str,
        track_id: int,
        action_frame: int,
        video_ref: Optional[str],
        force: bool = False,
    ) -> Optional[SubclassResult]:
        buf = self._buffers.get(track_id)
        if not buf:
            return None

        before_target = action_frame - self.window_before
        after_target = action_frame + self.window_after
        items = [(fi, lm) for fi, lm in buf
                 if before_target <= fi <= after_target]
        if not force and len(items) < self.window_before:
            return None
        if not items:
            return None
        items.sort(key=lambda p: p[0])
        points, scores = self._build_window(items)
        if points is None:
            return None

        pw = PoseWindow(
            points=points, scores=scores, fps=self.fps,
            frame_start=items[0][0], frame_end=items[-1][0],
            keypoint_format=KeypointFormat.MEDIAPIPE_13,
        )
        return self.recognizer.observe(
            parent_class=parent_class,
            pose_window=pw,
            source_track_id=track_id,
            video_ref=video_ref,
            session_id=self.session_id,
        )

    @staticmethod
    def _build_window(items: list) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        T = len(items)
        if T == 0:
            return None, None
        pts = np.zeros((T, 13, 2), dtype=np.float32)
        scs = np.zeros((T, 13), dtype=np.float32)
        for ti, (_, lm) in enumerate(items):
            for ki, name in enumerate(_LANDMARK_ORDER):
                v = lm.get(name)
                if v is None:
                    continue
                pts[ti, ki, 0] = float(v[0])
                pts[ti, ki, 1] = float(v[1])
                if len(v) > 2:
                    scs[ti, ki] = float(v[2])
                else:
                    scs[ti, ki] = 1.0
        return pts, scs
