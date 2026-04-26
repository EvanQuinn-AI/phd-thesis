"""Clinch + disocclusion handling for PvP (depth-free).

Detects clinch from heavy slot-bbox IoU OR detection collapse
(2 tracks → 1 detection while both tracks are alive). During clinch:
  - Kalman runs predict-only, no update from the merged detection.
  - FeatureBank updates suppressed (already gated in PvPTracker).
  - Action ownership returns ``None`` with reason ``clinch``.

On disocclusion (2 detections re-appear), the consumer should re-match
detections to slots using ``FeatureBank`` only (NOT the most recent
pre-clinch hist, which was contaminated). ``recover_assignment`` does this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tracking.config import DEFAULT, TrackingConfig
from tracking.features import PartExtractor
from tracking.kalman import iou
from tracking.masks import mask_iou


@dataclass
class ClinchState:
    active: bool = False
    started_frame: int = -1
    consecutive_high_iou: int = 0
    consecutive_collapse: int = 0
    uncertain_until_frame: int = -1
    last_recovery_confidence: float = 0.0


class ClinchDetector:
    def __init__(self, cfg: TrackingConfig = None):
        self.cfg = cfg or DEFAULT
        self.state = ClinchState()
        self._extractor = PartExtractor(self.cfg)

    def observe(
        self,
        frame_idx: int,
        slot_bboxes: dict,
        num_person_detections: int,
        person_dets: Optional[list[tuple]] = None,
        slot_masks: Optional[dict] = None,
        person_masks: Optional[list] = None,
    ) -> ClinchState:
        """Update clinch state. ``slot_bboxes`` is ``{tid: bbox or None}``.

        If ``person_dets`` is provided, two clearly-separated detections also
        force-exit the clinch (predicted-box IoU lags during predict-only
        mode and would otherwise keep the clinch active forever).

        Mask-aware operation: when ``slot_masks`` and/or ``person_masks`` are
        provided, mask-IoU replaces bbox-IoU for the corresponding signal.
        Bboxes overlap heavily as soon as fighters close range; masks overlap
        only when their pixels actually mix, so the clinch threshold is more
        meaningful with masks.
        """
        b1 = slot_bboxes.get("1")
        b2 = slot_bboxes.get("2")

        m1 = slot_masks.get("1") if slot_masks else None
        m2 = slot_masks.get("2") if slot_masks else None
        slot_masks_present = m1 is not None and m2 is not None

        if slot_masks_present:
            high_iou = mask_iou(m1, m2) > self.cfg.clinch_iou_thresh
        else:
            high_iou = (b1 is not None and b2 is not None
                        and iou(b1, b2) > self.cfg.clinch_iou_thresh)
        both_alive = b1 is not None and b2 is not None
        collapsed = both_alive and num_person_detections == 1

        if high_iou:
            self.state.consecutive_high_iou += 1
        else:
            self.state.consecutive_high_iou = 0

        if collapsed:
            self.state.consecutive_collapse += 1
        else:
            self.state.consecutive_collapse = 0

        # Detection-driven exit: two detections that are clearly apart.
        det_separated = False
        if person_masks is not None and len(person_masks) >= 2 \
                and person_masks[0] is not None and person_masks[1] is not None:
            det_separated = (
                mask_iou(person_masks[0], person_masks[1])
                < self.cfg.contamination_iou_thresh
            )
        elif person_dets is not None and len(person_dets) >= 2:
            det_separated = (
                iou(person_dets[0], person_dets[1]) < self.cfg.contamination_iou_thresh
            )

        should_enter = (
            self.state.consecutive_high_iou >= self.cfg.clinch_min_frames
            or self.state.consecutive_collapse >= self.cfg.clinch_min_frames
        )
        should_exit = (not high_iou and not collapsed) or det_separated

        if should_enter and not self.state.active and not det_separated:
            self.state.active = True
            self.state.started_frame = frame_idx
        elif should_exit and self.state.active:
            # Disocclusion: enter uncertain window during which consumers
            # should hold both ID hypotheses alive.
            self.state.active = False
            self.state.uncertain_until_frame = frame_idx + self.cfg.disocclusion_uncertain_window

        return self.state

    def is_uncertain(self, frame_idx: int) -> bool:
        return frame_idx <= self.state.uncertain_until_frame

    def recover_assignment(
        self,
        frame: np.ndarray,
        person_dets: list[tuple],
        landmarks_per_person: list,
        slot_banks: dict,
    ) -> dict:
        """Match (up to 2) detections to slots using FeatureBank scores only.

        Returns ``{tid: det_idx}``. If best-match confidence falls below
        ``disocclusion_min_match_conf``, the assignment is still returned but
        ``self.state.last_recovery_confidence`` flags it for the caller.
        """
        if len(person_dets) < 2 or "1" not in slot_banks or "2" not in slot_banks:
            return {}
        feats = [
            self._extractor.extract(frame, person_dets[j], landmarks_per_person[j])
            for j in range(len(person_dets))
        ]
        # Score every (slot, det) pair, take the configuration with the
        # smallest total chi^2.
        best: tuple[float, dict, float] | None = None
        for j1 in range(len(person_dets)):
            for j2 in range(len(person_dets)):
                if j1 == j2:
                    continue
                s1 = slot_banks["1"].score(feats[j1])
                s2 = slot_banks["2"].score(feats[j2])
                if s1 == float("inf") or s2 == float("inf"):
                    continue
                total = s1 + s2
                conf = (slot_banks["1"].confidence(feats[j1])
                        + slot_banks["2"].confidence(feats[j2])) / 2
                if best is None or total < best[0]:
                    best = (total, {"1": j1, "2": j2}, conf)
        if best is None:
            return {}
        self.state.last_recovery_confidence = best[2]
        return best[1]
