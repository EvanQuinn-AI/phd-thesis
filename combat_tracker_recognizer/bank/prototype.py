"""Prototype: per-(parent, subclass) running statistics + diagnostic exemplars."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


_VARIANCE_FLOOR = 1e-4
_EXEMPLAR_CAP = 20


@dataclass
class Prototype:
    parent_class: str
    subclass: str
    mean: np.ndarray  # (D,) float32
    m2: np.ndarray  # (D,) Welford M2
    exemplar_count: int = 0
    exemplar_embeddings: deque = field(default_factory=lambda: deque(maxlen=_EXEMPLAR_CAP))
    version: int = 0
    encoder_version: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_first(cls, parent: str, subclass: str, embedding: np.ndarray,
                   encoder_version: str) -> "Prototype":
        d = embedding.shape[0]
        proto = cls(
            parent_class=parent,
            subclass=subclass,
            mean=np.zeros(d, dtype=np.float32),
            m2=np.zeros(d, dtype=np.float32),
            encoder_version=encoder_version,
        )
        proto.update(embedding)
        return proto

    def update(self, embedding: np.ndarray) -> None:
        """Welford online mean + variance accumulation."""
        x = embedding.astype(np.float32, copy=False)
        self.exemplar_count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.exemplar_count
        delta2 = x - self.mean
        self.m2 = self.m2 + delta * delta2
        self.exemplar_embeddings.append(x.copy())
        self.version += 1
        self.updated_at = datetime.utcnow()

    def variance(self) -> np.ndarray:
        if self.exemplar_count < 2:
            return np.full_like(self.mean, _VARIANCE_FLOOR)
        return np.maximum(self.m2 / (self.exemplar_count - 1), _VARIANCE_FLOOR)

    def variance_trace(self) -> float:
        return float(self.variance().sum())

    def should_split(self, threshold: float) -> bool:
        return self.variance_trace() > threshold and self.exemplar_count >= 4

    def distance(self, embedding: np.ndarray) -> float:
        x = embedding.astype(np.float32, copy=False)
        if self.exemplar_count >= 5:
            inv_var = 1.0 / self.variance()
            diff = x - self.mean
            return float(np.sqrt(np.sum(diff * diff * inv_var) / self.mean.shape[0]))
        # Cosine distance for sparse banks.
        denom = (np.linalg.norm(self.mean) * np.linalg.norm(x)) + 1e-8
        cos = float(np.dot(self.mean, x) / denom)
        return 1.0 - cos


def merge_welford(p_a: Prototype, p_b: Prototype, new_subclass: str) -> Prototype:
    """Combine two prototypes' running stats into a new prototype.

    Uses Chan's parallel-merge formula for Welford's algorithm.
    """
    n_a, n_b = p_a.exemplar_count, p_b.exemplar_count
    if n_a == 0:
        out = Prototype(
            parent_class=p_a.parent_class, subclass=new_subclass,
            mean=p_b.mean.copy(), m2=p_b.m2.copy(),
            exemplar_count=n_b, encoder_version=p_a.encoder_version,
        )
    elif n_b == 0:
        out = Prototype(
            parent_class=p_a.parent_class, subclass=new_subclass,
            mean=p_a.mean.copy(), m2=p_a.m2.copy(),
            exemplar_count=n_a, encoder_version=p_a.encoder_version,
        )
    else:
        n = n_a + n_b
        delta = p_b.mean - p_a.mean
        mean = p_a.mean + delta * (n_b / n)
        m2 = p_a.m2 + p_b.m2 + (delta * delta) * (n_a * n_b / n)
        out = Prototype(
            parent_class=p_a.parent_class, subclass=new_subclass,
            mean=mean.astype(np.float32), m2=m2.astype(np.float32),
            exemplar_count=n, encoder_version=p_a.encoder_version,
        )
    # Concatenate exemplar buffers (capped).
    for e in list(p_a.exemplar_embeddings) + list(p_b.exemplar_embeddings):
        out.exemplar_embeddings.append(e)
    out.version = max(p_a.version, p_b.version) + 1
    return out
