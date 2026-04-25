"""PvP online tracker.

Extends ``update_two_person_ids`` (Combat Sports Automation PvP/gpu-version/app.py:429)
with motion (Kalman) + pose-indexed part-histogram ReID. Returns a dict
shaped exactly like the existing ``tracked`` dict so the Streamlit app's
downstream CSV/overlay code is untouched:

    {"1": {"box": (x1,y1,x2,y2), "hist": np.ndarray, "action_counts": {...}},
     "2": {...}}

The legacy ``hist`` field is populated with a whole-bbox histogram so any
downstream code that still reads it keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from tracking.anchoring import AnchoredFighter, IdentityAnchor
from tracking.base import FeatureBank, Track
from tracking.config import DEFAULT, TrackingConfig
from tracking.features import PartExtractor
from tracking.kalman import BBoxKalman, iou
from tracking.occlusion import ClinchDetector


def _legacy_full_hist(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


@dataclass
class PvPSlot:
    track_id: str
    bbox: Optional[tuple] = None
    kalman: Optional[BBoxKalman] = None
    feature_bank: FeatureBank = None
    last_hist: Optional[np.ndarray] = None
    age_since_seen: int = 0
    front_foot: Optional[str] = None
    start_region: Optional[str] = None


class PvPTracker:
    """Two-slot tracker that supersedes ``update_two_person_ids``.

    Anchoring: pass a pre-populated ``IdentityAnchor`` to ``__init__``, or
    feed the first N frames through ``observe_for_anchor`` and call
    ``finalize_anchor`` before regular ``update`` calls.
    """

    def __init__(
        self,
        cfg: Optional[TrackingConfig] = None,
        anchor: Optional[IdentityAnchor] = None,
    ):
        self.cfg = cfg or DEFAULT
        self.extractor = PartExtractor(self.cfg)
        self.slots: dict[str, PvPSlot] = {
            "1": PvPSlot(track_id="1", feature_bank=FeatureBank()),
            "2": PvPSlot(track_id="2", feature_bank=FeatureBank()),
        }
        self._anchor = anchor or IdentityAnchor(self.cfg)
        self.anchored = False
        self.clinch = ClinchDetector(self.cfg)
        self._frame_idx = 0
        self.last_assignment_log: list[tuple] = []  # for debug/eval overlay

    # ---- Anchoring -----------------------------------------------------

    def observe_for_anchor(
        self,
        frame: np.ndarray,
        person_dets: list[tuple],
        landmarks_per_person: list[Optional[dict]],
    ) -> bool:
        """Returns True once the anchoring window is full."""
        self._anchor.observe(frame, person_dets, landmarks_per_person)
        return self._anchor.is_full

    def finalize_anchor(self) -> None:
        fighters = self._anchor.finalize()
        for tid, fighter in fighters.items():
            slot = self.slots[tid]
            slot.feature_bank = fighter.feature_bank
            slot.front_foot = fighter.front_foot
            slot.start_region = fighter.start_region
            if fighter.bbox_history:
                slot.bbox = fighter.bbox_history[-1]
                slot.kalman = BBoxKalman(slot.bbox, process_noise=1.5)
        self.anchored = True

    # ---- Per-frame update ---------------------------------------------

    def _kalman_predict(self) -> dict[str, Optional[tuple]]:
        out: dict[str, Optional[tuple]] = {}
        for tid, slot in self.slots.items():
            if slot.kalman is None:
                out[tid] = None
                continue
            out[tid] = slot.kalman.predict()
        return out

    @staticmethod
    def _box_iou_matrix(predicted: list[tuple], dets: list[tuple]) -> np.ndarray:
        if not predicted or not dets:
            return np.zeros((len(predicted), len(dets)))
        m = np.zeros((len(predicted), len(dets)), dtype=np.float32)
        for i, p in enumerate(predicted):
            for j, d in enumerate(dets):
                m[i, j] = iou(p, d) if p is not None else 0.0
        return m

    def _greedy_iou_assign(
        self,
        slot_ids: list[str],
        predicted_boxes: list[Optional[tuple]],
        dets: list[tuple],
    ) -> tuple[dict[str, int], set[int]]:
        present_predicted = [p for p in predicted_boxes if p is not None]
        present_idx = [i for i, p in enumerate(predicted_boxes) if p is not None]
        m = self._box_iou_matrix(present_predicted, dets)
        assignments: dict[str, int] = {}
        used_dets: set[int] = set()
        used_slots: set[int] = set()
        flat = [(m[i, j], i, j) for i in range(m.shape[0]) for j in range(m.shape[1])]
        flat.sort(reverse=True)
        for score, i, j in flat:
            if score < self.cfg.iou_match_threshold:
                break
            if i in used_slots or j in used_dets:
                continue
            assignments[slot_ids[present_idx[i]]] = j
            used_slots.add(i)
            used_dets.add(j)
        return assignments, used_dets

    def _reid_assign(
        self,
        unassigned_slot_ids: list[str],
        candidate_indices: list[int],
        dets: list[tuple],
        landmarks_per_person: list[Optional[dict]],
        frame: np.ndarray,
    ) -> dict[str, int]:
        if not unassigned_slot_ids or not candidate_indices:
            return {}
        scored: list[tuple[float, str, int]] = []
        for j in candidate_indices:
            feats = self.extractor.extract(frame, dets[j], landmarks_per_person[j])
            for tid in unassigned_slot_ids:
                slot = self.slots[tid]
                s = slot.feature_bank.score(feats) if slot.feature_bank else float("inf")
                scored.append((s, tid, j))
        scored.sort(key=lambda x: x[0])
        out: dict[str, int] = {}
        used_slots: set[str] = set()
        used_dets: set[int] = set()
        for s, tid, j in scored:
            if s == float("inf"):
                continue
            if tid in used_slots or j in used_dets:
                continue
            out[tid] = j
            used_slots.add(tid)
            used_dets.add(j)
        return out

    def _commit_assignment(
        self,
        tid: str,
        bbox: tuple,
        landmarks: Optional[dict],
        frame: np.ndarray,
        suppress_bank_update: bool,
    ) -> None:
        slot = self.slots[tid]
        if slot.kalman is None:
            slot.kalman = BBoxKalman(bbox, process_noise=1.5)
        else:
            slot.kalman.update(bbox)
        slot.bbox = slot.kalman.bbox()
        slot.age_since_seen = 0
        slot.last_hist = _legacy_full_hist(frame, slot.bbox)
        if not suppress_bank_update:
            feats = self.extractor.extract(frame, slot.bbox, landmarks)
            conf = self.extractor.mean_landmark_visibility(landmarks) if landmarks else 0.5
            if conf >= self.cfg.pose_conf_for_bank_update or not landmarks:
                slot.feature_bank.add_features(feats, conf)

    def update(
        self,
        frame: np.ndarray,
        person_dets: list[tuple],
        landmarks_per_person: Optional[list[Optional[dict]]] = None,
    ) -> dict:
        """Return a dict shaped like the legacy ``tracked`` dict.

        ``landmarks_per_person`` may be ``None`` to signal "no pose data this
        frame," which forces the legacy whole-bbox histogram fallback.
        """
        if landmarks_per_person is None:
            landmarks_per_person = [None] * len(person_dets)

        # Auto-anchor if we haven't yet but have enough observations queued.
        if not self.anchored:
            self.observe_for_anchor(frame, person_dets, landmarks_per_person)
            if self._anchor.is_full:
                self.finalize_anchor()

        slot_ids = ["1", "2"]
        predicted = self._kalman_predict()
        predicted_boxes = [predicted[tid] for tid in slot_ids]

        # Clinch detection runs on PREDICTED boxes (so it triggers even when
        # the detector has merged the two fighters), but also takes the raw
        # detections so it can force-exit when two clearly-separate boxes
        # reappear.
        clinch_state = self.clinch.observe(
            self._frame_idx,
            {tid: predicted[tid] for tid in slot_ids},
            num_person_detections=len(person_dets),
            person_dets=person_dets,
        )
        was_clinched_last_frame = self._is_post_clinch_recovery_frame()
        in_clinch = clinch_state.active

        if in_clinch:
            # Predict-only: do NOT update Kalman or banks from merged/overlapping
            # detections. Hold both tracks in place; let them drift gracefully.
            for tid in slot_ids:
                slot = self.slots[tid]
                slot.age_since_seen += 1
                if slot.kalman is not None:
                    slot.bbox = predicted[tid]
            for tid, slot in self.slots.items():
                if slot.age_since_seen > self.cfg.max_age_frames:
                    slot.bbox = None
                    slot.kalman = None
            self.last_assignment_log.append(({}, True))
            self._frame_idx += 1
            return self.as_legacy_tracked_dict()

        if was_clinched_last_frame and len(person_dets) >= 2:
            # Disocclusion recovery: re-anchor slots to detections by feature-bank
            # match alone (NOT by IoU against drifted Kalman predictions).
            slot_banks = {tid: self.slots[tid].feature_bank for tid in slot_ids}
            recovery = self.clinch.recover_assignment(
                frame, person_dets, landmarks_per_person, slot_banks,
            )
            if recovery:
                # Reset Kalman state at the recovered detection so subsequent
                # IoU passes don't snap back to the wrong fighter.
                for tid, det_idx in recovery.items():
                    self.slots[tid].kalman = BBoxKalman(person_dets[det_idx], process_noise=1.5)
                    self.slots[tid].bbox = person_dets[det_idx]
                # Apply the start_region sanity check below.

        # Pass 1: IoU assignment.
        iou_assignments, used = self._greedy_iou_assign(slot_ids, predicted_boxes, person_dets)

        # Pass 2: ReID for unassigned.
        unassigned = [tid for tid in slot_ids if tid not in iou_assignments]
        candidate_idx = [j for j in range(len(person_dets)) if j not in used]
        reid_assignments = self._reid_assign(unassigned, candidate_idx, person_dets,
                                             landmarks_per_person, frame)

        all_assignments = {**iou_assignments, **reid_assignments}

        # start_region sanity check: if both slots are assigned and their detections
        # land in the wrong half of the frame relative to their anchored start_region,
        # consult the feature banks. The banks act as the tiebreaker.
        if len(all_assignments) == 2 and self.anchored:
            a_idx = all_assignments["1"]
            b_idx = all_assignments["2"]
            self._maybe_swap_on_region_violation(
                a_idx, b_idx, person_dets, landmarks_per_person, frame, all_assignments,
            )

        # Detect whether two slots overlap heavily (contamination gate).
        suppress_bank_update_flag = False
        if len(all_assignments) == 2:
            a, b = list(all_assignments.values())
            if iou(person_dets[a], person_dets[b]) > self.cfg.contamination_iou_thresh:
                suppress_bank_update_flag = True

        for tid, det_idx in all_assignments.items():
            self._commit_assignment(
                tid,
                person_dets[det_idx],
                landmarks_per_person[det_idx],
                frame,
                suppress_bank_update=suppress_bank_update_flag,
            )

        # Slots that got no detection: age + Kalman-only.
        for tid in slot_ids:
            if tid in all_assignments:
                continue
            slot = self.slots[tid]
            slot.age_since_seen += 1
            if slot.kalman is not None:
                slot.bbox = predicted[tid]

        # Drop slots that have been lost too long. We keep the slot id but null its bbox.
        for tid, slot in self.slots.items():
            if slot.age_since_seen > self.cfg.max_age_frames:
                slot.bbox = None
                slot.kalman = None

        self.last_assignment_log.append((all_assignments, suppress_bank_update_flag))
        if len(self.last_assignment_log) > 240:
            self.last_assignment_log = self.last_assignment_log[-240:]
        self._frame_idx += 1

        return self.as_legacy_tracked_dict()

    def _is_post_clinch_recovery_frame(self) -> bool:
        """First frame after a clinch ends: previous log entry's clinch flag set, current isn't."""
        if not self.last_assignment_log:
            return False
        prev_clinch = bool(self.last_assignment_log[-1][1] is True
                           and not self.last_assignment_log[-1][0])
        return prev_clinch and not self.clinch.state.active

    def _maybe_swap_on_region_violation(
        self,
        a_idx: int,
        b_idx: int,
        person_dets: list[tuple],
        landmarks_per_person: list,
        frame: np.ndarray,
        assignments: dict,
    ) -> None:
        """If both detections sit in the wrong half relative to anchored start_region,
        compare bank scores under both orderings and pick the lower total chi^2."""
        slot_1 = self.slots["1"]
        slot_2 = self.slots["2"]
        if slot_1.start_region is None or slot_2.start_region is None:
            return
        # Mid-x of the frame.
        mid_x = frame.shape[1] / 2.0
        a_left = ((person_dets[a_idx][0] + person_dets[a_idx][2]) / 2.0) < mid_x
        b_left = ((person_dets[b_idx][0] + person_dets[b_idx][2]) / 2.0) < mid_x
        # Slot 1 anchored left, slot 2 anchored right by IdentityAnchor convention.
        violates = (slot_1.start_region == "left_half" and not a_left) and \
                   (slot_2.start_region == "right_half" and not b_left)
        if not violates:
            return
        # Score both orderings against the banks.
        feats_a = self.extractor.extract(frame, person_dets[a_idx], landmarks_per_person[a_idx])
        feats_b = self.extractor.extract(frame, person_dets[b_idx], landmarks_per_person[b_idx])
        current_score = slot_1.feature_bank.score(feats_a) + slot_2.feature_bank.score(feats_b)
        swapped_score = slot_1.feature_bank.score(feats_b) + slot_2.feature_bank.score(feats_a)
        if swapped_score < current_score:
            assignments["1"], assignments["2"] = b_idx, a_idx

    def as_legacy_tracked_dict(self) -> dict:
        """Shape-compatible with ``update_two_person_ids``'s ``tracked`` dict."""
        return {
            tid: {
                "box": slot.bbox,
                "hist": slot.last_hist,
                "action_counts": {},
                "front_foot": slot.front_foot,
                "start_region": slot.start_region,
                "clinch": self.clinch.state.active,
            }
            for tid, slot in self.slots.items()
        }
