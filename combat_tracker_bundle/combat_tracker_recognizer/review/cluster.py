"""HDBSCAN clustering of unlabeled clips."""

from __future__ import annotations

from typing import Optional

import numpy as np
from hdbscan import HDBSCAN

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.config import ReviewConfig
from combat_tracker_recognizer.types import Cluster, Clip


def _stack_embeddings(clips: list[Clip]) -> np.ndarray:
    return np.stack([c.embedding.vector for c in clips]).astype(np.float64)


def _suggest_labels(
    bank: Optional[PrototypeBank],
    parent: str,
    embedding: np.ndarray,
    max_n: int,
) -> list[tuple[str, float]]:
    if bank is None:
        return []
    return bank.match(parent, embedding)[:max_n]


def cluster_unknowns(
    clips: list[Clip],
    parent: Optional[str],
    config: ReviewConfig,
    bank: Optional[PrototypeBank] = None,
) -> list[Cluster]:
    """Cluster ``clips`` and return a list of ``Cluster`` records.

    HDBSCAN noise points (label -1) become singleton clusters so each
    can be individually labeled or discarded — preserving the
    one-clip-one-decision contract from the plan.
    """
    if parent is not None:
        clips = [c for c in clips if c.parent_class == parent]
    if not clips:
        return []

    X = _stack_embeddings(clips)

    out: list[Cluster] = []

    if len(clips) >= max(2, config.min_cluster_size):
        # HDBSCAN doesn't support metric="cosine" with the default
        # algorithm; convert to euclidean on L2-normalised vectors,
        # which is order-equivalent.
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        hdb = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            metric="euclidean",
            allow_single_cluster=True,
        )
        labels = hdb.fit_predict(Xn)
    else:
        # Too few clips for HDBSCAN: every clip is a singleton.
        labels = np.full(len(clips), -1)

    next_id = 0

    # Real clusters
    cluster_ids = sorted(set(int(lab) for lab in labels if lab != -1))
    for cl in cluster_ids:
        member_idx = [i for i, lab in enumerate(labels) if int(lab) == cl]
        member_embs = X[member_idx]
        mean = member_embs.mean(axis=0)
        # Medoid = clip closest to mean.
        d = np.linalg.norm(member_embs - mean, axis=1)
        medoid_local = int(np.argmin(d))
        medoid_clip = clips[member_idx[medoid_local]]
        intra_d = float(np.mean([np.linalg.norm(member_embs[i] - mean)
                                 for i in range(len(member_idx))]))
        sugg = _suggest_labels(bank, medoid_clip.parent_class,
                               medoid_clip.embedding.vector,
                               config.max_suggested_labels)
        out.append(Cluster(
            id=next_id,
            parent_class=medoid_clip.parent_class,
            exemplar_clip_id=int(medoid_clip.id),
            member_clip_ids=[int(clips[i].id) for i in member_idx],
            size=len(member_idx),
            suggested_labels=sugg,
            intra_distance_mean=intra_d,
        ))
        next_id += 1

    # Noise points → singleton clusters
    for i, lab in enumerate(labels):
        if int(lab) != -1:
            continue
        c = clips[i]
        sugg = _suggest_labels(bank, c.parent_class, c.embedding.vector,
                               config.max_suggested_labels)
        out.append(Cluster(
            id=next_id,
            parent_class=c.parent_class,
            exemplar_clip_id=int(c.id),
            member_clip_ids=[int(c.id)],
            size=1,
            suggested_labels=sugg,
            intra_distance_mean=0.0,
        ))
        next_id += 1

    return out
