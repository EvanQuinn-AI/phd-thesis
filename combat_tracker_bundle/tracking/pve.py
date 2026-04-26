"""PvE tracker: single person + bag, with impact attribution.

Replaces the bbox-overlap-with-bag heuristic in
``Combat Sports Automation/gpu-version/app.py:check_overlap`` (line 312)
with: track the person, track the bag, and attribute landed impacts using
wrist trajectory + bag swing-state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tracking.base import Track
from tracking.kalman import BBoxKalman, iou


BAG_VELOCITY_STRUCK_THRESHOLD = 6.0  # px/frame, EMA centroid velocity magnitude
BAG_VELOCITY_SWING_THRESHOLD = 2.0


@dataclass
class ImpactEvent:
    action_id: int
    landed: bool
    impact_point: Optional[tuple] = None
    confidence: float = 0.0
    reason: str = ""


class PvETracker:
    """Single-person + bag tracker.

    Public surface:
        update(frame_idx, person_dets, bag_dets) -> dict of state.
    Returned dict keys: ``person_track``, ``bag_track``, ``bag_state``.
    """

    def __init__(self, max_age: int = 30):
        self.person: Optional[Track] = None
        self.bag: Optional[Track] = None
        self._person_kalman: Optional[BBoxKalman] = None
        self._bag_kalman: Optional[BBoxKalman] = None
        self.max_age = max_age
        self._bag_vel_ema = 0.0
        self.bag_state = "resting"
        self._secondary_person_streak = 0
        self.warnings: list[str] = []

    @staticmethod
    def _largest(boxes: list) -> Optional[tuple]:
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def _update_person(self, frame_idx: int, person_dets: list[tuple]) -> None:
        if len(person_dets) > 1:
            self._secondary_person_streak += 1
            if self._secondary_person_streak == 10:
                self.warnings.append(
                    f"frame {frame_idx}: persistent multiple persons; consider PvP mode"
                )
        else:
            self._secondary_person_streak = 0

        primary = self._largest(person_dets)

        if self.person is None:
            if primary is None:
                return
            self.person = Track(track_id="1")
            self._person_kalman = BBoxKalman(primary, process_noise=1.0)
            self.person.mark_seen(frame_idx, primary)
            return

        predicted = self._person_kalman.predict()
        if primary is None:
            self.person.bbox = predicted
            self.person.mark_missed()
            return
        if iou(predicted, primary) < 0.05 and self.person.age_since_seen == 0:
            # Sudden jump; trust detection but reset velocity.
            self._person_kalman = BBoxKalman(primary, process_noise=1.0)
        else:
            self._person_kalman.update(primary)
        self.person.mark_seen(frame_idx, self._person_kalman.bbox())

    def _update_bag(self, frame_idx: int, bag_dets: list[tuple]) -> None:
        primary = self._largest(bag_dets)

        if self.bag is None:
            if primary is None:
                return
            self.bag = Track(track_id="bag")
            self._bag_kalman = BBoxKalman(primary, process_noise=0.2)
            self.bag.mark_seen(frame_idx, primary)
            return

        prev_centre = self._centre(self.bag.bbox)
        self._bag_kalman.predict()
        if primary is not None:
            self._bag_kalman.update(primary)
            self.bag.mark_seen(frame_idx, self._bag_kalman.bbox())
        else:
            self.bag.bbox = self._bag_kalman.bbox()
            self.bag.mark_missed()

        new_centre = self._centre(self.bag.bbox)
        instantaneous = float(np.hypot(new_centre[0] - prev_centre[0], new_centre[1] - prev_centre[1]))
        # EMA on velocity magnitude.
        self._bag_vel_ema = 0.6 * self._bag_vel_ema + 0.4 * instantaneous
        if self._bag_vel_ema > BAG_VELOCITY_STRUCK_THRESHOLD:
            self.bag_state = "struck"
        elif self._bag_vel_ema > BAG_VELOCITY_SWING_THRESHOLD:
            self.bag_state = "swinging"
        else:
            self.bag_state = "resting"

    @staticmethod
    def _centre(bbox: tuple) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(
        self,
        frame_idx: int,
        person_dets: list[tuple],
        bag_dets: list[tuple],
    ) -> dict:
        self._update_person(frame_idx, person_dets)
        self._update_bag(frame_idx, bag_dets)

        if self.person and self.person.is_lost(self.max_age):
            self.person = None
            self._person_kalman = None
        if self.bag and self.bag.is_lost(self.max_age):
            self.bag = None
            self._bag_kalman = None
            self.bag_state = "resting"

        return {
            "person_track": self.person,
            "bag_track": self.bag,
            "bag_state": self.bag_state,
            "bag_velocity_ema": self._bag_vel_ema,
        }


class ImpactAttributor:
    """Attribute landed/missed labels to action events using wrist trajectory + bag state.

    The existing PvE app already detects events as frame-ranges via a debounce
    state machine. This attributor decorates each event with landed/missed and
    a confidence score, replacing the per-frame bbox-overlap heuristic.
    """

    def __init__(self, temporal_window: int = 3):
        self.temporal_window = temporal_window
        self._bag_state_history: list[tuple[int, str]] = []

    def record_bag_state(self, frame_idx: int, state: str) -> None:
        self._bag_state_history.append((frame_idx, state))
        if len(self._bag_state_history) > 240:
            self._bag_state_history = self._bag_state_history[-240:]

    def _struck_within(self, centre_frame: int) -> bool:
        for fi, st in self._bag_state_history:
            if abs(fi - centre_frame) <= self.temporal_window and st == "struck":
                return True
        return False

    def attribute(
        self,
        action_id: int,
        action_frame: int,
        terminal_keypoint_xy: Optional[tuple[int, int]],
        bag_bbox: Optional[tuple],
    ) -> ImpactEvent:
        if bag_bbox is None or terminal_keypoint_xy is None:
            return ImpactEvent(action_id=action_id, landed=False, confidence=0.0,
                               reason="no_bag_or_keypoint")
        x, y = terminal_keypoint_xy
        x1, y1, x2, y2 = bag_bbox
        spatial = (x1 <= x <= x2) and (y1 <= y <= y2)
        temporal = self._struck_within(action_frame)
        if spatial and temporal:
            return ImpactEvent(action_id=action_id, landed=True, impact_point=(x, y),
                               confidence=0.9, reason="spatial+temporal")
        if spatial:
            return ImpactEvent(action_id=action_id, landed=True, impact_point=(x, y),
                               confidence=0.55, reason="spatial_only")
        if temporal:
            return ImpactEvent(action_id=action_id, landed=True, confidence=0.4,
                               reason="temporal_only")
        return ImpactEvent(action_id=action_id, landed=False, confidence=0.8,
                           reason="no_spatial_no_temporal")
