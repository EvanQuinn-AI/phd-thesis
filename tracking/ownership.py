"""Action ownership: who threw what, who got hit.

Replaces ``find_action_owner`` (Combat Sports Automation PvP/gpu-version/app.py:498)
which used "person whose box contains the action centroid."

This implementation uses a kinematic-chain best-fit on pose landmarks,
falling back to the legacy centroid rule when pose data is unavailable
(preserves baseline behaviour for the ablation table).

Clinch suppression: if a ``ClinchDetector`` is provided and active, returns
an unattributed event with ``reason="clinch"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tracking.kalman import iou
from tracking.masks import mask_overlap_fraction


_TERMINAL_KEYPOINT = {
    "punch": ("left_wrist", "right_wrist"),
    "cross": ("left_wrist", "right_wrist"),
    "hook": ("left_wrist", "right_wrist"),
    "kick": ("left_ankle", "right_ankle"),
    "kick-knee": ("left_ankle", "right_ankle"),
}


@dataclass
class AttributedAction:
    action_id: int
    action_class: str
    owner_id: Optional[str]
    target_id: Optional[str]
    confidence: float
    reason: str
    method: str  # "mask_iou" | "kinematic" | "centroid_fallback" | "unattributed"


def _action_centre(action_box: tuple) -> tuple[int, int]:
    x1, y1, x2, y2 = action_box
    return (x1 + x2) // 2, (y1 + y2) // 2


def _box_contains(box: tuple, x: int, y: int) -> bool:
    if box is None:
        return False
    bx1, by1, bx2, by2 = box
    return bx1 <= x <= bx2 and by1 <= y <= by2


def _denormalise(landmark: tuple, frame_w: int, frame_h: int) -> tuple[int, int, float]:
    x_norm, y_norm, vis = landmark
    return int(x_norm * frame_w), int(y_norm * frame_h), float(vis)


def _terminal_distance(
    landmarks: dict,
    keypoint_names: tuple[str, str],
    frame_w: int,
    frame_h: int,
    target_xy: tuple[int, int],
    visibility_thresh: float,
) -> Optional[tuple[float, tuple[int, int]]]:
    """Min distance from any of ``keypoint_names`` (visible above threshold) to ``target_xy``.

    Returns ``(distance, terminal_xy)`` or ``None`` if no keypoint passes.
    """
    best: Optional[tuple[float, tuple[int, int]]] = None
    for name in keypoint_names:
        lm = landmarks.get(name) if landmarks else None
        if lm is None:
            continue
        x, y, v = _denormalise(lm, frame_w, frame_h)
        if v < visibility_thresh:
            continue
        d = float(np.hypot(x - target_xy[0], y - target_xy[1]))
        if best is None or d < best[0]:
            best = (d, (x, y))
    return best


class ActionOwnership:
    """Assign (owner, target) to an action event.

    Stores no per-frame state itself; takes the current tracker snapshot
    and pose history at call time.
    """

    def __init__(self, visibility_thresh: float = 0.4, clinch_detector=None):
        self.visibility_thresh = visibility_thresh
        self.clinch = clinch_detector

    def assign(
        self,
        action_id: int,
        action_class: str,
        action_box: tuple,
        frame_idx: int,
        frame_size: tuple[int, int],
        tracks: dict,
        landmarks_per_track: dict,
        bag_box: Optional[tuple] = None,
        action_mask: Optional[np.ndarray] = None,
        masks_per_track: Optional[dict] = None,
        bag_mask: Optional[np.ndarray] = None,
    ) -> AttributedAction:
        """Assign owner + target.

        Three-tier rule, in precedence order:

            1. **mask_iou**  — owner = arg max_k |action_mask ∩ mask_k| / |action_mask|.
               Requires the action mask AND at least two track masks.
            2. **kinematic** — owner = track whose terminal keypoint is closest
               to the action centre. Requires pose landmarks for at least one
               track.
            3. **centroid_fallback** — owner = track whose bbox uniquely
               contains the action-box centroid.

        Args:
            tracks: ``{tid: {"box": ...}}`` – matches the legacy ``tracked`` shape.
            landmarks_per_track: ``{tid: landmark_dict or None}``.
            bag_box: optional bag bbox for PvE-style action targeting.
            action_mask: optional segmentation mask for the action detection.
            masks_per_track: optional ``{tid: mask}`` for each fighter.
            bag_mask: optional bag mask.
        """
        # Suppress during clinch.
        if self.clinch is not None and self.clinch.state.active:
            return AttributedAction(
                action_id=action_id, action_class=action_class,
                owner_id=None, target_id=None, confidence=0.0,
                reason="clinch", method="unattributed",
            )

        action_centre = _action_centre(action_box)
        frame_w, frame_h = frame_size
        keypoint_names = _TERMINAL_KEYPOINT.get(action_class, ("left_wrist", "right_wrist"))

        owner_id: Optional[str]
        method: str
        confidence: float
        terminal_xy: Optional[tuple[int, int]] = action_centre

        # Tier 1: mask-IoU best-fit.
        mask_owner = None
        if action_mask is not None and masks_per_track:
            scored = []
            for tid, m in masks_per_track.items():
                if m is None:
                    continue
                frac = mask_overlap_fraction(action_mask, m)
                scored.append((frac, tid))
            scored.sort(reverse=True)
            if scored and scored[0][0] > 0.0:
                mask_owner = scored[0]

        if mask_owner is not None:
            owner_id = mask_owner[1]
            confidence = float(mask_owner[0])  # fraction in [0, 1]
            method = "mask_iou"
        else:
            # Tier 2: kinematic-chain rule.
            best: Optional[tuple[float, str, tuple[int, int]]] = None
            for tid, lm in landmarks_per_track.items():
                if not lm:
                    continue
                d = _terminal_distance(lm, keypoint_names, frame_w, frame_h,
                                       action_centre, self.visibility_thresh)
                if d is None:
                    continue
                distance, kp_xy = d
                if best is None or distance < best[0]:
                    best = (distance, tid, kp_xy)

            if best is not None:
                distance, owner_id, terminal_xy = best
                diag = float(np.hypot(frame_w, frame_h))
                confidence = max(0.0, 1.0 - distance / (0.25 * diag))
                method = "kinematic"
            else:
                # Tier 3: legacy centroid containment.
                containers = [tid for tid, t in tracks.items()
                              if _box_contains(t.get("box"), *action_centre)]
                if len(containers) == 1:
                    owner_id = containers[0]
                    confidence = 0.5
                    method = "centroid_fallback"
                    terminal_xy = action_centre
                else:
                    return AttributedAction(
                        action_id=action_id, action_class=action_class,
                        owner_id=None, target_id=None, confidence=0.0,
                        reason="ambiguous_centroid", method="unattributed",
                    )

        # Target inference. With masks, target = candidate whose mask the
        # action mask overlaps most (excluding the owner). Without masks,
        # fall back to bbox-IoU + inverse distance to centroid.
        target_id: Optional[str] = None
        if action_mask is not None and (masks_per_track or bag_mask is not None):
            scored: list[tuple[float, str]] = []
            for tid, m in (masks_per_track or {}).items():
                if tid == owner_id or m is None:
                    continue
                scored.append((mask_overlap_fraction(action_mask, m), tid))
            if bag_mask is not None and "bag" != owner_id:
                scored.append((mask_overlap_fraction(action_mask, bag_mask), "bag"))
            scored.sort(reverse=True)
            if scored and scored[0][0] > 0:
                target_id = scored[0][1]
        else:
            diag = float(np.hypot(frame_w, frame_h))
            candidates: list[tuple[float, str]] = []
            for tid, t in tracks.items():
                if tid == owner_id:
                    continue
                box = t.get("box")
                if box is None:
                    continue
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                dist = float(np.hypot(cx - action_centre[0], cy - action_centre[1]))
                score = iou(action_box, box) + max(0.0, 1.0 - dist / diag)
                candidates.append((score, tid))
            if bag_box is not None:
                cx = (bag_box[0] + bag_box[2]) / 2
                cy = (bag_box[1] + bag_box[3]) / 2
                dist = float(np.hypot(cx - action_centre[0], cy - action_centre[1]))
                score = iou(action_box, bag_box) + max(0.0, 1.0 - dist / diag)
                candidates.append((score, "bag"))
            candidates.sort(reverse=True)
            if candidates and candidates[0][0] > 0:
                target_id = candidates[0][1]

        return AttributedAction(
            action_id=action_id, action_class=action_class,
            owner_id=owner_id, target_id=target_id, confidence=confidence,
            reason="ok", method=method,
        )
