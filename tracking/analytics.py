"""Per-fighter analytics for the qualitative eval.

Aggregates frame-level events into thesis-defensible offensive /
defensive / movement metrics. The accumulator is fed by the eval runner
once per frame and emits a final JSON summary plus optional per-frame
rows for plotting.

Metrics tracked per slot id (``"1"``, ``"2"``):

  Offensive
    - throws_total, throws_by_class (cross, hook, kick, ...)
    - landed_total, landed_by_class           (mask-overlap or bbox-overlap)
    - hit_rate = landed / throws
    - throws_per_minute (uses fps + total frames)

  Defensive
    - guards_active_frames (high-guard + low-guard combined)
    - time_in_clinch_frames
    - shots_received_total, shots_received_by_class

  Movement / kinematics
    - travel_distance_px       (cumulative bbox-centroid path length)
    - peak_centroid_speed_pps  (px/sec)
    - mean_centroid_speed_pps
    - mean_wrist_speed_pps     (left + right average; from pose if available)

  Engagement
    - mean_distance_between_fighters_px
    - frames_within_strike_range  (centroid distance < threshold)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


_STRIKE_RANGE_FRAC = 0.35  # frac of frame diagonal


@dataclass
class _SlotState:
    last_centroid: Optional[tuple[float, float]] = None
    last_wrist_l: Optional[tuple[float, float]] = None
    last_wrist_r: Optional[tuple[float, float]] = None
    centroid_path_px: float = 0.0
    centroid_speeds_pps: list[float] = field(default_factory=list)
    wrist_speeds_pps: list[float] = field(default_factory=list)
    throws_by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    landed_by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    received_by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    guards_active_frames: int = 0
    time_in_clinch_frames: int = 0


class FighterAnalytics:
    """Accumulator. Call ``observe_frame`` once per frame."""

    def __init__(self, fps: float, frame_size: tuple[int, int]):
        self.fps = max(fps, 1.0)
        self.frame_w, self.frame_h = frame_size
        self._slots: dict[str, _SlotState] = {
            "1": _SlotState(),
            "2": _SlotState(),
        }
        self._engagement_distances: list[float] = []
        self._frames_within_strike_range: int = 0
        self._total_frames: int = 0
        self._strike_range_px: float = _STRIKE_RANGE_FRAC * float(np.hypot(self.frame_w, self.frame_h))

    # ---- Per-frame ingestion ------------------------------------------

    def observe_frame(
        self,
        slot_bboxes: dict[str, Optional[tuple]],
        slot_landmarks: dict[str, Optional[dict]],
        clinch_active: bool,
        action_classes_present: list[str],
    ) -> None:
        """Update centroid travel + speed for both slots, engagement
        distance, clinch counter, and guard counter."""
        self._total_frames += 1
        dt = 1.0 / self.fps

        for tid in ("1", "2"):
            box = slot_bboxes.get(tid)
            slot = self._slots[tid]
            if clinch_active:
                slot.time_in_clinch_frames += 1
            if box is not None:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                centroid = (float(cx), float(cy))
                if slot.last_centroid is not None:
                    step = float(np.hypot(centroid[0] - slot.last_centroid[0],
                                          centroid[1] - slot.last_centroid[1]))
                    slot.centroid_path_px += step
                    slot.centroid_speeds_pps.append(step / dt)
                slot.last_centroid = centroid

            lm = slot_landmarks.get(tid)
            if lm:
                w_l = lm.get("left_wrist")
                w_r = lm.get("right_wrist")
                for prev_attr, cur in (("last_wrist_l", w_l), ("last_wrist_r", w_r)):
                    if cur is None or cur[2] < 0.4:
                        continue
                    cur_xy = (cur[0] * self.frame_w, cur[1] * self.frame_h)
                    prev = getattr(slot, prev_attr)
                    if prev is not None:
                        step = float(np.hypot(cur_xy[0] - prev[0],
                                              cur_xy[1] - prev[1]))
                        slot.wrist_speeds_pps.append(step / dt)
                    setattr(slot, prev_attr, cur_xy)

        # Engagement distance.
        b1 = slot_bboxes.get("1"); b2 = slot_bboxes.get("2")
        if b1 is not None and b2 is not None:
            cx1 = (b1[0] + b1[2]) / 2; cy1 = (b1[1] + b1[3]) / 2
            cx2 = (b2[0] + b2[2]) / 2; cy2 = (b2[1] + b2[3]) / 2
            d = float(np.hypot(cx1 - cx2, cy1 - cy2))
            self._engagement_distances.append(d)
            if d < self._strike_range_px:
                self._frames_within_strike_range += 1

        # Guard frames: any high/low-guard class present this frame.
        guards_present = any(c in action_classes_present
                             for c in ("high-guard", "low-guard"))
        if guards_present:
            for slot in self._slots.values():
                slot.guards_active_frames += 1

    # ---- Per-action ingestion -----------------------------------------

    def record_action_thrown(self, owner_id: Optional[str], action_class: str) -> None:
        if owner_id in self._slots:
            self._slots[owner_id].throws_by_class[action_class] += 1

    def record_action_landed(
        self,
        owner_id: Optional[str],
        target_id: Optional[str],
        action_class: str,
    ) -> None:
        if owner_id in self._slots:
            self._slots[owner_id].landed_by_class[action_class] += 1
        if target_id in self._slots and target_id != owner_id:
            self._slots[target_id].received_by_class[action_class] += 1

    # ---- Reporting ----------------------------------------------------

    def _slot_summary(self, tid: str) -> dict:
        s = self._slots[tid]
        throws_total = sum(s.throws_by_class.values())
        landed_total = sum(s.landed_by_class.values())
        received_total = sum(s.received_by_class.values())
        minutes = max(self._total_frames / self.fps / 60.0, 1e-6)
        return {
            "throws_total": throws_total,
            "throws_by_class": dict(s.throws_by_class),
            "landed_total": landed_total,
            "landed_by_class": dict(s.landed_by_class),
            "hit_rate": (landed_total / throws_total) if throws_total else 0.0,
            "throws_per_minute": throws_total / minutes,
            "shots_received_total": received_total,
            "shots_received_by_class": dict(s.received_by_class),
            "guards_active_frames": s.guards_active_frames,
            "time_in_clinch_frames": s.time_in_clinch_frames,
            "time_in_clinch_seconds": s.time_in_clinch_frames / self.fps,
            "travel_distance_px": s.centroid_path_px,
            "mean_centroid_speed_pps": (float(np.mean(s.centroid_speeds_pps))
                                        if s.centroid_speeds_pps else 0.0),
            "peak_centroid_speed_pps": (float(np.max(s.centroid_speeds_pps))
                                        if s.centroid_speeds_pps else 0.0),
            "mean_wrist_speed_pps": (float(np.mean(s.wrist_speeds_pps))
                                     if s.wrist_speeds_pps else 0.0),
        }

    def summary(self) -> dict:
        return {
            "fps": self.fps,
            "total_frames": self._total_frames,
            "duration_seconds": self._total_frames / self.fps,
            "fighters": {
                tid: self._slot_summary(tid) for tid in ("1", "2")
            },
            "engagement": {
                "mean_distance_between_fighters_px": (
                    float(np.mean(self._engagement_distances))
                    if self._engagement_distances else 0.0
                ),
                "frames_within_strike_range": self._frames_within_strike_range,
                "strike_range_threshold_px": self._strike_range_px,
            },
        }
