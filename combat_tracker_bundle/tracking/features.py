"""Pose-keypoint-indexed colour-histogram extractor.

Regions are sampled from MediaPipe's 13-landmark subset that the existing
pose_analytics.py uses. Landmarks are passed in as a dict
``{name: (x_norm, y_norm, visibility)}`` matching that module's output schema.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from tracking.config import DEFAULT, TrackingConfig


# Region -> list of landmark names that bound it. Missing landmarks fall back
# to the bbox itself for that region.
_REGION_LANDMARKS = {
    "gloves_L": ["left_wrist"],
    "gloves_R": ["right_wrist"],
    "trunks": ["left_hip", "right_hip"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "head": ["nose"],
}


class PartExtractor:
    def __init__(self, cfg: Optional[TrackingConfig] = None):
        self.cfg = cfg or DEFAULT

    @staticmethod
    def _landmark_pixel(lm: tuple, frame_w: int, frame_h: int) -> tuple[int, int, float]:
        x_norm, y_norm, vis = lm
        return int(x_norm * frame_w), int(y_norm * frame_h), float(vis)

    def _glove_box(self, wrist_xy: tuple[int, int], bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        r = max(8, int(0.06 * h))
        cx, cy = wrist_xy
        return (max(x1, cx - r), max(y1, cy - r), min(x2, cx + r), min(y2, cy + r))

    def _trunks_box(self, hips: list[tuple[int, int]], bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        cx = int(np.mean([p[0] for p in hips]))
        cy = int(np.mean([p[1] for p in hips]))
        rx = max(20, int(0.18 * (x2 - x1)))
        ry = max(15, int(0.10 * h))
        return (max(x1, cx - rx), max(y1, cy - ry), min(x2, cx + rx), min(y2, cy + ry))

    def _torso_box(self, pts: list[tuple[int, int]], bbox: tuple) -> tuple:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (max(bbox[0], min(xs)), max(bbox[1], min(ys)),
                min(bbox[2], max(xs)), min(bbox[3], max(ys)))

    def _head_box(self, nose_xy: tuple[int, int], bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)
        r = max(15, int(0.08 * h))
        cx, cy = nose_xy
        return (max(x1, cx - r), max(y1, cy - r), min(x2, cx + r), min(y2, cy + r))

    def _hist(self, frame: np.ndarray, region: tuple,
              mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = region
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return None
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist_mask = None
        if mask is not None:
            mask_patch = mask[y1:y2, x1:x2]
            if mask_patch.shape == hsv.shape[:2] and mask_patch.any():
                hist_mask = (mask_patch.astype(np.uint8) * 255)
            else:
                # Region is entirely outside the instance mask -> region not
                # informative, skip.
                return None
        hist = cv2.calcHist(
            [hsv], [0, 1], hist_mask,
            [self.cfg.hist_h_bins, self.cfg.hist_s_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def extract(
        self,
        frame: np.ndarray,
        bbox: tuple,
        landmarks: Optional[dict] = None,
        mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Return ``{region: histogram}`` dict. Missing keypoints → region omitted.

        ``mask`` is an optional binary instance mask (same shape as ``frame``
        in HxW). When provided, each region's pixel sample is intersected
        with the mask, removing background and other-fighter contamination
        before the HSV histogram is computed. With ``mask=None`` the legacy
        bbox-only behaviour is preserved exactly.
        """
        h, w = frame.shape[:2]
        out: dict[str, np.ndarray] = {}

        if landmarks is None:
            # Fall back to whole-bbox histogram only.
            hist = self._hist(frame, tuple(bbox), mask=mask)
            if hist is not None:
                out["torso"] = hist
            return out

        vt = self.cfg.landmark_visibility_thresh
        pts = {}
        for name, lm in landmarks.items():
            x, y, v = self._landmark_pixel(lm, w, h)
            if v >= vt:
                pts[name] = (x, y)

        if "left_wrist" in pts:
            box = self._glove_box(pts["left_wrist"], bbox)
            hist = self._hist(frame, box, mask=mask)
            if hist is not None:
                out["gloves_L"] = hist
        if "right_wrist" in pts:
            box = self._glove_box(pts["right_wrist"], bbox)
            hist = self._hist(frame, box, mask=mask)
            if hist is not None:
                out["gloves_R"] = hist
        hips = [pts[n] for n in ("left_hip", "right_hip") if n in pts]
        if len(hips) >= 1:
            box = self._trunks_box(hips, bbox)
            hist = self._hist(frame, box, mask=mask)
            if hist is not None:
                out["trunks"] = hist
        torso_pts = [pts[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip") if n in pts]
        if len(torso_pts) >= 3:
            box = self._torso_box(torso_pts, bbox)
            hist = self._hist(frame, box, mask=mask)
            if hist is not None:
                out["torso"] = hist
        if "nose" in pts:
            box = self._head_box(pts["nose"], bbox)
            hist = self._hist(frame, box, mask=mask)
            if hist is not None:
                out["head"] = hist
        return out

    def mean_landmark_visibility(self, landmarks: dict) -> float:
        if not landmarks:
            return 0.0
        return float(np.mean([lm[2] for lm in landmarks.values()]))
