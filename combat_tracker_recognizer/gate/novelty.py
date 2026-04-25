"""NoveltyGate: routes embeddings to KNOWN / AMBIGUOUS / UNKNOWN / NOISE."""

from __future__ import annotations

from typing import Optional

import numpy as np

from combat_tracker_recognizer.bank.bank import PrototypeBank
from combat_tracker_recognizer.config import GateConfig
from combat_tracker_recognizer.types import GateDecision


class NoveltyGate:
    def __init__(self, bank: PrototypeBank, config: GateConfig):
        self.bank = bank
        self.cfg = config

    def route(
        self,
        parent_class: str,
        embedding: np.ndarray,
    ) -> tuple[GateDecision, Optional[str], float, list[tuple[str, float]]]:
        """Return ``(decision, subclass_or_None, confidence, top_matches)``.

        Confidence in [0, 1]. For KNOWN: ``1 - d1``. For AMBIGUOUS:
        ``1 - d1`` capped to 0.5. For UNKNOWN/NOISE: 0.
        """
        if float(np.linalg.norm(embedding)) < self.cfg.noise_magnitude_floor:
            return GateDecision.NOISE, None, 0.0, []

        matches = self.bank.match(parent_class, embedding)
        if not matches:
            return GateDecision.UNKNOWN, None, 0.0, []

        d1 = matches[0][1]
        d2 = matches[1][1] if len(matches) > 1 else float("inf")

        if d1 < self.cfg.known_distance_threshold and (d2 / max(d1, 1e-6)) >= self.cfg.min_margin_ratio:
            return GateDecision.KNOWN, matches[0][0], max(0.0, 1.0 - d1), matches

        if d1 < self.cfg.ambiguous_distance_threshold:
            return GateDecision.AMBIGUOUS, None, min(0.5, max(0.0, 1.0 - d1)), matches

        return GateDecision.UNKNOWN, None, 0.0, matches
