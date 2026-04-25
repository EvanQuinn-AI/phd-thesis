"""Identity anchoring for PvP.

Locks two fighter identities (``"1"`` and ``"2"``) using priors from the
first N frames before the fight gets messy:

  - start_region: which half of the frame each fighter starts in
  - front_foot: ``left`` | ``right`` (neutral; replaces orthodox/southpaw to
    match the thesis's existing vocabulary)
  - height_px: bbox-height EMA over the window (no depth normalisation)
  - feature_bank: populated from every frame where pose visibility passes
    the configured threshold

No corner-colour detection: thesis has no corner concept yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tracking.base import FeatureBank
from tracking.config import DEFAULT, TrackingConfig
from tracking.features import PartExtractor


@dataclass
class AnchoredFighter:
    track_id: str
    start_region: str  # "left_half" | "right_half"
    front_foot: Optional[str] = None  # "left" | "right" | None
    height_px: float = 0.0
    feature_bank: FeatureBank = None
    bbox_history: list[tuple] = field(default_factory=list)

    def mean_bbox_centre_x(self) -> float:
        if not self.bbox_history:
            return 0.0
        return float(np.mean([(b[0] + b[2]) / 2 for b in self.bbox_history]))


class IdentityAnchor:
    def __init__(self, cfg: Optional[TrackingConfig] = None):
        self.cfg = cfg or DEFAULT
        self.extractor = PartExtractor(self.cfg)
        self._frames: list[dict] = []
        self.frame_w: Optional[int] = None
        self.frame_h: Optional[int] = None

    @property
    def window(self) -> int:
        return self.cfg.anchor_window_frames

    @property
    def is_full(self) -> bool:
        return len(self._frames) >= self.window

    def observe(
        self,
        frame: np.ndarray,
        person_dets: list[tuple],
        landmarks_per_person: list[Optional[dict]],
    ) -> None:
        """Buffer a frame's worth of observations. Discards extras beyond top-2 by area."""
        if self.is_full:
            return
        if self.frame_w is None:
            self.frame_h, self.frame_w = frame.shape[:2]
        # Sort by area desc and keep top 2 to match the existing PvP convention.
        pairs = sorted(
            zip(person_dets, landmarks_per_person),
            key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]),
            reverse=True,
        )[:2]
        self._frames.append({"frame": frame, "pairs": pairs})

    @staticmethod
    def _front_foot(landmarks: Optional[dict], facing_right: bool) -> Optional[str]:
        """Foot whose x is closer to the opponent (facing direction)."""
        if not landmarks:
            return None
        lf = landmarks.get("left_ankle")
        rf = landmarks.get("right_ankle")
        if lf is None or rf is None:
            return None
        if lf[2] < 0.4 or rf[2] < 0.4:
            return None
        # Forward = larger x if facing_right, else smaller x.
        if facing_right:
            return "left" if lf[0] > rf[0] else "right"
        return "left" if lf[0] < rf[0] else "right"

    def finalize(self) -> dict[str, AnchoredFighter]:
        """Aggregate observations into two anchored fighters keyed by ``"1"`` and ``"2"``."""
        if not self._frames:
            return {}

        # Per-frame, assign the leftmost detection to slot A and the rightmost to slot B,
        # then aggregate. This is more robust than the existing "top 2 by area" approach
        # for anchoring (size assumptions break under perspective).
        bank_a, bank_b = FeatureBank(), FeatureBank()
        boxes_a: list[tuple] = []
        boxes_b: list[tuple] = []
        landmarks_a: list[Optional[dict]] = []
        landmarks_b: list[Optional[dict]] = []

        for record in self._frames:
            pairs = record["pairs"]
            if len(pairs) < 2:
                continue
            ordered = sorted(pairs, key=lambda p: (p[0][0] + p[0][2]) / 2)
            (box_l, lm_l), (box_r, lm_r) = ordered[0], ordered[1]
            boxes_a.append(box_l)
            boxes_b.append(box_r)
            landmarks_a.append(lm_l)
            landmarks_b.append(lm_r)
            f_l = self.extractor.extract(record["frame"], box_l, lm_l)
            f_r = self.extractor.extract(record["frame"], box_r, lm_r)
            conf_l = self.extractor.mean_landmark_visibility(lm_l) if lm_l else 0.5
            conf_r = self.extractor.mean_landmark_visibility(lm_r) if lm_r else 0.5
            if conf_l >= self.cfg.pose_conf_for_bank_update or not lm_l:
                bank_a.add_features(f_l, conf_l)
            if conf_r >= self.cfg.pose_conf_for_bank_update or not lm_r:
                bank_b.add_features(f_r, conf_r)

        if not boxes_a or not boxes_b:
            return {}

        height_a = float(np.mean([b[3] - b[1] for b in boxes_a]))
        height_b = float(np.mean([b[3] - b[1] for b in boxes_b]))
        # Vote front-foot from frames with valid landmarks.
        votes_a: list[str] = []
        votes_b: list[str] = []
        for lm in landmarks_a:
            ff = self._front_foot(lm, facing_right=True)
            if ff:
                votes_a.append(ff)
        for lm in landmarks_b:
            ff = self._front_foot(lm, facing_right=False)
            if ff:
                votes_b.append(ff)
        front_a = max(set(votes_a), key=votes_a.count) if votes_a else None
        front_b = max(set(votes_b), key=votes_b.count) if votes_b else None

        fighter_1 = AnchoredFighter(
            track_id="1",
            start_region="left_half",
            front_foot=front_a,
            height_px=height_a,
            feature_bank=bank_a,
            bbox_history=boxes_a[-5:],
        )
        fighter_2 = AnchoredFighter(
            track_id="2",
            start_region="right_half",
            front_foot=front_b,
            height_px=height_b,
            feature_bank=bank_b,
            bbox_history=boxes_b[-5:],
        )
        return {"1": fighter_1, "2": fighter_2}
