"""Track + FeatureBank primitives shared by PvE and PvP trackers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tracking.config import DEFAULT, TrackingConfig


@dataclass
class Track:
    """A single tracked entity (person or bag).

    bbox is (x1, y1, x2, y2) in pixel coords.
    kalman_state is opaque to consumers — it lives inside the Kalman wrapper.
    """

    track_id: str
    bbox: Optional[tuple] = None
    last_seen_frame: int = -1
    age_since_seen: int = 0
    kalman_state: object = None
    feature_bank: "FeatureBank" = None
    extras: dict = field(default_factory=dict)

    def mark_seen(self, frame_idx: int, bbox: tuple) -> None:
        self.bbox = bbox
        self.last_seen_frame = frame_idx
        self.age_since_seen = 0

    def mark_missed(self) -> None:
        self.age_since_seen += 1

    def is_lost(self, max_age: int) -> bool:
        return self.age_since_seen > max_age


def _chi2_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric chi-squared distance between two normalised histograms."""
    return 0.5 * float(np.sum(((a - b) ** 2) / (a + b + eps)))


class FeatureBank:
    """Per-identity rolling store of part histograms.

    Top-K samples per region by pose confidence. New samples evict the
    lowest-confidence stored sample only when the bank is full.
    """

    def __init__(self, capacity: int = None, region_weights: dict = None):
        cfg: TrackingConfig = DEFAULT
        self.capacity = capacity if capacity is not None else cfg.feature_bank_size
        self.region_weights = region_weights if region_weights is not None else dict(cfg.reid_region_weights)
        # region -> list of (confidence, histogram)
        self._store: dict[str, list[tuple[float, np.ndarray]]] = {}

    def add(self, region: str, hist: np.ndarray, confidence: float) -> None:
        if hist is None:
            return
        bucket = self._store.setdefault(region, [])
        bucket.append((float(confidence), hist.astype(np.float32, copy=True)))
        if len(bucket) > self.capacity:
            bucket.sort(key=lambda x: x[0], reverse=True)
            del bucket[self.capacity:]

    def add_features(self, features: dict, confidence: float) -> None:
        for region, hist in features.items():
            self.add(region, hist, confidence)

    def has_region(self, region: str) -> bool:
        return region in self._store and len(self._store[region]) > 0

    def best_distance(self, region: str, candidate: np.ndarray) -> Optional[float]:
        bucket = self._store.get(region)
        if not bucket or candidate is None:
            return None
        return min(_chi2_distance(candidate, h) for _, h in bucket)

    def score(self, candidate_features: dict) -> float:
        """Lower is better. Weighted mean χ² across regions present in both bank and candidate."""
        weighted_sum = 0.0
        weight_total = 0.0
        for region, hist in candidate_features.items():
            d = self.best_distance(region, hist)
            if d is None:
                continue
            w = self.region_weights.get(region, 0.0)
            if w <= 0:
                continue
            weighted_sum += w * d
            weight_total += w
        if weight_total == 0:
            return float("inf")
        return weighted_sum / weight_total

    def confidence(self, candidate_features: dict) -> float:
        """Match confidence in [0, 1]. 1 - normalised χ². Used by Phase 5 disocclusion gate."""
        s = self.score(candidate_features)
        if s == float("inf"):
            return 0.0
        # χ² distances on H+S 16x16 hists are typically 0..2; clamp.
        return max(0.0, 1.0 - min(s, 1.0))
