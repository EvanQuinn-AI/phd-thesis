"""PrototypeBank: keyed by (parent_class, subclass), holds 1+ prototypes per key."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from combat_tracker_recognizer.bank.prototype import (
    Prototype,
    merge_welford,
)
from combat_tracker_recognizer.encoders.protocol import Encoder


class PrototypeBank:
    def __init__(self) -> None:
        # (parent, subclass) -> list[Prototype]. Multi-prototype to handle
        # high-variance subclasses post-split.
        self._store: dict[tuple[str, str], list[Prototype]] = {}

    # ---- Mutation ------------------------------------------------------

    def add(self, parent: str, subclass: str, embedding: np.ndarray,
            encoder_version: str = "") -> None:
        key = (parent, subclass)
        bucket = self._store.get(key)
        if not bucket:
            self._store[key] = [Prototype.from_first(parent, subclass, embedding, encoder_version)]
            return
        # Update the prototype currently nearest to the new sample.
        nearest = min(bucket, key=lambda p: p.distance(embedding))
        nearest.update(embedding)

    def split(self, parent: str, subclass: str) -> None:
        """k-means(k=2) over the union of exemplar buffers; redistribute samples."""
        key = (parent, subclass)
        bucket = self._store.get(key)
        if not bucket:
            raise KeyError(f"no prototypes for {key}")
        exemplars: list[np.ndarray] = []
        for p in bucket:
            exemplars.extend(p.exemplar_embeddings)
        if len(exemplars) < 4:
            raise ValueError(f"need >=4 exemplars to split; have {len(exemplars)}")
        X = np.stack(exemplars)
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(X), size=2, replace=False)
        c1, c2 = X[idxs[0]].copy(), X[idxs[1]].copy()
        for _ in range(20):
            d1 = np.linalg.norm(X - c1, axis=1)
            d2 = np.linalg.norm(X - c2, axis=1)
            assign = (d2 < d1)
            new_c1 = X[~assign].mean(axis=0) if (~assign).any() else c1
            new_c2 = X[assign].mean(axis=0) if assign.any() else c2
            if np.allclose(new_c1, c1) and np.allclose(new_c2, c2):
                break
            c1, c2 = new_c1, new_c2
        # Materialise as two prototypes under suffixed subclass names.
        names = [f"{subclass}__a", f"{subclass}__b"]
        new_protos = []
        encoder_version = bucket[0].encoder_version
        for grp_idx, mask in enumerate([~assign, assign]):
            if not mask.any():
                continue
            samples = X[mask]
            proto = Prototype.from_first(parent, names[grp_idx], samples[0], encoder_version)
            for s in samples[1:]:
                proto.update(s)
            new_protos.append(proto)
        del self._store[key]
        for p in new_protos:
            self._store[(parent, p.subclass)] = [p]

    def merge(self, parent: str, subclass_a: str, subclass_b: str, new_subclass: str) -> None:
        ka, kb = (parent, subclass_a), (parent, subclass_b)
        if ka not in self._store or kb not in self._store:
            raise KeyError(f"merge needs both subclasses present; have {sorted(self._store)}")
        # If either side has multiple prototypes, collapse them first.
        bucket_a = self._store.pop(ka)
        bucket_b = self._store.pop(kb)
        merged_a = bucket_a[0]
        for p in bucket_a[1:]:
            merged_a = merge_welford(merged_a, p, subclass_a)
        merged_b = bucket_b[0]
        for p in bucket_b[1:]:
            merged_b = merge_welford(merged_b, p, subclass_b)
        out = merge_welford(merged_a, merged_b, new_subclass)
        self._store[(parent, new_subclass)] = [out]

    def rename(self, parent: str, old_subclass: str, new_subclass: str) -> None:
        old_key = (parent, old_subclass)
        if old_key not in self._store:
            raise KeyError(old_key)
        bucket = self._store.pop(old_key)
        for p in bucket:
            p.subclass = new_subclass
        self._store[(parent, new_subclass)] = bucket

    def remove(self, parent: str, subclass: str) -> None:
        self._store.pop((parent, subclass), None)

    def rebuild_from_clips(self, clips: Iterable, encoder: Encoder) -> None:
        """Re-encode every clip and rebuild the bank in place.

        ``clips`` items must have ``.parent_class``, ``.pose``, and a label
        association via ``getattr(c, 'subclass', None)`` (Phase 2 ClipStore
        provides this when fetching labeled clips).
        """
        self._store.clear()
        for c in clips:
            subclass = getattr(c, "subclass", None)
            if subclass is None:
                continue
            emb = encoder.encode(c.pose)
            self.add(c.parent_class, subclass, emb, encoder_version=encoder.version)

    # ---- Query ---------------------------------------------------------

    def match(self, parent: str, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Return ``[(subclass, distance), ...]`` sorted ascending. Parent-scoped."""
        out: list[tuple[str, float]] = []
        for (p, sc), bucket in self._store.items():
            if p != parent:
                continue
            best = min(bucket, key=lambda pr: pr.distance(embedding))
            out.append((sc, best.distance(embedding)))
        out.sort(key=lambda t: t[1])
        return out

    def all_subclasses(self, parent: Optional[str] = None) -> list[tuple[str, str]]:
        if parent is None:
            return sorted(self._store.keys())
        return sorted(k for k in self._store if k[0] == parent)

    def num_prototypes(self) -> int:
        return sum(len(v) for v in self._store.values())

    def get(self, parent: str, subclass: str) -> list[Prototype]:
        return list(self._store.get((parent, subclass), []))
