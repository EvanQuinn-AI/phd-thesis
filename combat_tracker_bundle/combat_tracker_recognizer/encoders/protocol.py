"""Encoder Protocol: enforces a stable interface across encoder implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from combat_tracker_recognizer.types import PoseWindow


@runtime_checkable
class Encoder(Protocol):
    version: str
    embedding_dim: int

    def encode(self, window: PoseWindow) -> np.ndarray: ...

    def encode_batch(self, windows: list[PoseWindow]) -> np.ndarray: ...
