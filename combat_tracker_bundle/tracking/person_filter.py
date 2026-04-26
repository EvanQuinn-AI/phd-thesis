"""Pre-filter person detections to drop referees, coaches, and background.

Two complementary signals:

1. **Relative size**: a detection whose bbox area is smaller than
   ``min_area_ratio`` of the largest person bbox in the frame is dropped.
   Background people in the depth plane are always smaller in image
   coords. Threshold at 0.30 by default — empirically distinguishes
   ringside/cage-side from in-ring fighters without being too aggressive.

2. **Motion energy**: per detection, track centroid over a rolling
   window. If the variance in centroid position is below
   ``stationary_variance_thresh`` (in pixels^2) over a full window of
   observations, mark stationary. Referees / corner staff stand still;
   fighters move continuously. Persistent-identity matching across
   frames is done with simple greedy IoU over the previous frame's
   bboxes — this is a *filter* not a tracker, so it doesn't need
   long-horizon ID stability.

Output is split into ``(kept, filtered)`` so the eval overlay can render
filtered detections in a different colour, giving a thesis-defensible
"this is what was suppressed and why" record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tracking.kalman import iou


@dataclass
class FilterReason:
    bbox: tuple
    reason: str  # "small_area" | "stationary" | "kept"


@dataclass
class _Tracklet:
    """Lightweight per-bbox motion record. NOT a full track — only used
    to score stationarity over the recent window."""

    bbox: tuple
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=20))
    age: int = 0

    def update(self, bbox: tuple) -> None:
        self.bbox = bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.centroid_history.append((cx, cy))
        self.age += 1

    def is_stationary(self, variance_thresh: float, min_obs: int) -> bool:
        if len(self.centroid_history) < min_obs:
            return False
        arr = np.asarray(self.centroid_history, dtype=np.float64)
        v = float(arr.var(axis=0).sum())
        return v < variance_thresh


class PersonFilter:
    """Stateful detection-level filter. Call once per frame.

    Args:
        min_area_ratio: drop detections whose area is below this ratio of
            the largest person bbox in the same frame. Default 0.30.
        stationary_variance_thresh: centroid-variance threshold (px^2)
            below which a tracklet is considered stationary. Default 200.
        stationary_min_observations: tracklet age in frames before the
            stationary check is allowed to fire. Default 12.
        carry_over_iou: minimum IoU to associate a current-frame bbox
            with an existing tracklet. Default 0.4.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.30,
        stationary_variance_thresh: float = 200.0,
        stationary_min_observations: int = 12,
        carry_over_iou: float = 0.4,
    ):
        self.min_area_ratio = min_area_ratio
        self.stationary_variance_thresh = stationary_variance_thresh
        self.stationary_min_observations = stationary_min_observations
        self.carry_over_iou = carry_over_iou
        self._tracklets: list[_Tracklet] = []

    @staticmethod
    def _area(b: tuple) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def _associate(self, bboxes: list[tuple]) -> dict[int, _Tracklet]:
        """Greedy IoU match each bbox to an existing tracklet; create new ones for unmatched.

        Returns ``{bbox_idx: tracklet}``.
        """
        used = set()
        out: dict[int, _Tracklet] = {}
        for i, bb in enumerate(bboxes):
            best = (-1.0, None)
            for t in self._tracklets:
                if id(t) in used:
                    continue
                s = iou(bb, t.bbox)
                if s > best[0]:
                    best = (s, t)
            if best[0] >= self.carry_over_iou and best[1] is not None:
                used.add(id(best[1]))
                best[1].update(bb)
                out[i] = best[1]
            else:
                t = _Tracklet(bbox=bb)
                t.update(bb)
                self._tracklets.append(t)
                out[i] = t
        # Drop tracklets that haven't been touched this frame and are old.
        self._tracklets = [t for t in self._tracklets
                           if id(t) in used or t.age < 5 or t in out.values()]
        return out

    def filter(
        self,
        person_dets: list[tuple],
        landmarks_per_person: Optional[list] = None,
    ) -> tuple[list[tuple], list, list[FilterReason]]:
        """Return ``(kept_dets, kept_landmarks, decisions)``.

        Filters only activate when ``len(person_dets) > 2`` — the explicit
        "more than two persons in the frame" case where a referee, coach, or
        bystander has been detected. With <= 2 detections we assume both are
        fighters (the 2-slot tracker enforces that anyway) and pass through
        unchanged. This avoids the failure mode where a briefly-stationary
        fighter is dropped during a pause between exchanges.
        """
        if not person_dets:
            return [], (landmarks_per_person or []), []

        if landmarks_per_person is None:
            landmarks_per_person = [None] * len(person_dets)
        elif len(landmarks_per_person) != len(person_dets):
            raise ValueError("landmarks_per_person length must match person_dets")

        # Always update tracklets so stationarity history is up to date even
        # when filtering is bypassed.
        bbox_to_tracklet = self._associate(person_dets)

        if len(person_dets) <= 2:
            return list(person_dets), list(landmarks_per_person), [
                FilterReason(bbox=bb, reason="kept") for bb in person_dets
            ]

        max_area = max(self._area(b) for b in person_dets)
        decisions: list[FilterReason] = []
        kept: list[tuple] = []
        kept_landmarks: list = []
        for i, bb in enumerate(person_dets):
            area = self._area(bb)
            if max_area > 0 and (area / max_area) < self.min_area_ratio:
                decisions.append(FilterReason(bbox=bb, reason="small_area"))
                continue
            t = bbox_to_tracklet[i]
            if t.is_stationary(self.stationary_variance_thresh,
                               self.stationary_min_observations):
                decisions.append(FilterReason(bbox=bb, reason="stationary"))
                continue
            decisions.append(FilterReason(bbox=bb, reason="kept"))
            kept.append(bb)
            kept_landmarks.append(landmarks_per_person[i])
        # If filtering removed everything (suspicious), keep top-2 by area as a safety.
        if not kept and len(person_dets) >= 2:
            ordered = sorted(enumerate(person_dets), key=lambda p: -self._area(p[1]))[:2]
            kept = [person_dets[i] for i, _ in ordered]
            kept_landmarks = [landmarks_per_person[i] for i, _ in ordered]
            for i, _ in ordered:
                decisions[i] = FilterReason(bbox=person_dets[i], reason="kept")
        return kept, kept_landmarks, decisions
