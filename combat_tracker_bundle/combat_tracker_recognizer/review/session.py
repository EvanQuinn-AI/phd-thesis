"""ReviewSession: human-in-the-loop labeling.

All bank-mutating operations auto-snapshot before they execute.
``rollback`` restores the bank to its state at session open (or the
most recent snapshot, whichever is later).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Optional

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.config import ReviewConfig
from combat_tracker_recognizer.review.cluster import cluster_unknowns
from combat_tracker_recognizer.store import ClipStore
from combat_tracker_recognizer.types import Cluster


@dataclass
class _Change:
    op: str
    payload: dict = field(default_factory=dict)


class ReviewSession:
    def __init__(
        self,
        session_id: str,
        clipstore: ClipStore,
        bank: PrototypeBank,
        config: ReviewConfig,
        encoder_version: str,
    ):
        self.session_id = session_id
        self.store = clipstore
        self.bank = bank
        self.cfg = config
        self.encoder_version = encoder_version
        self._changes: list[_Change] = []
        self._cluster_cache: Optional[list[Cluster]] = None
        # Take an entry snapshot so rollback always has a target.
        self._entry_snapshot_id = self._snapshot("session-open")

    def _snapshot(self, note: str) -> int:
        return self.store.save_bank_snapshot(
            pickle.dumps(self.bank._store), self.encoder_version, note=note,
        )

    def list_clusters(self, parent: Optional[str] = None) -> list[Cluster]:
        clips = self.store.get_unlabeled(parent=parent)
        self._cluster_cache = cluster_unknowns(clips, parent, self.cfg, bank=self.bank)
        return self._cluster_cache

    def _ensure_clusters(self) -> list[Cluster]:
        if self._cluster_cache is None:
            self.list_clusters()
        assert self._cluster_cache is not None
        return self._cluster_cache

    def get_cluster(self, cluster_id: int) -> Cluster:
        for c in self._ensure_clusters():
            if c.id == cluster_id:
                return c
        raise KeyError(f"cluster {cluster_id} not found")

    def label_cluster(
        self,
        cluster_id: int,
        subclass: str,
        parent_class: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        cluster = self.get_cluster(cluster_id)
        parent = parent_class or cluster.parent_class
        for clip_id in cluster.member_clip_ids:
            clip = self.store.get_clip(clip_id)
            self.store.label_clip(clip_id, subclass, parent, labeled_by="review",
                                  session_id=self.session_id, note=note)
            self.bank.add(parent, subclass, clip.embedding.vector,
                          encoder_version=self.encoder_version)
        self._changes.append(_Change(op="label_cluster",
                                     payload={"cluster_id": cluster_id,
                                              "subclass": subclass,
                                              "parent_class": parent}))
        # Invalidate cluster cache; labelled clips drop out of the unlabeled set.
        self._cluster_cache = None

    def discard_cluster(self, cluster_id: int, note: Optional[str] = None) -> None:
        cluster = self.get_cluster(cluster_id)
        for clip_id in cluster.member_clip_ids:
            self.store.discard_clip(clip_id, note=note, labeled_by="review")
        self._changes.append(_Change(op="discard_cluster",
                                     payload={"cluster_id": cluster_id}))
        self._cluster_cache = None

    def relabel_clip(
        self,
        clip_id: int,
        new_subclass: str,
        parent_class: Optional[str] = None,
    ) -> None:
        clip = self.store.get_clip(clip_id)
        parent = parent_class or clip.parent_class
        self._snapshot(f"pre-relabel-{clip_id}")
        self.store.relabel_clip(clip_id, new_subclass, parent, labeled_by="review")
        self.bank.add(parent, new_subclass, clip.embedding.vector,
                      encoder_version=self.encoder_version)
        self._changes.append(_Change(op="relabel_clip",
                                     payload={"clip_id": clip_id,
                                              "new_subclass": new_subclass}))
        self._cluster_cache = None

    def merge_clusters(self, cluster_id_a: int, cluster_id_b: int) -> int:
        """Merge two clusters in the in-memory view. The merged cluster
        replaces the two with the lower of the two ids. Returns the new id."""
        a = self.get_cluster(cluster_id_a)
        b = self.get_cluster(cluster_id_b)
        if a.parent_class != b.parent_class:
            raise ValueError("merge requires same parent_class")
        new_id = min(a.id, b.id)
        merged = Cluster(
            id=new_id, parent_class=a.parent_class,
            exemplar_clip_id=a.exemplar_clip_id,
            member_clip_ids=list(a.member_clip_ids) + list(b.member_clip_ids),
            size=a.size + b.size,
            suggested_labels=a.suggested_labels,
            intra_distance_mean=(a.intra_distance_mean + b.intra_distance_mean) / 2,
        )
        cur = self._ensure_clusters()
        cur = [c for c in cur if c.id not in (a.id, b.id)]
        cur.append(merged)
        cur.sort(key=lambda c: c.id)
        self._cluster_cache = cur
        self._changes.append(_Change(op="merge_clusters",
                                     payload={"a": cluster_id_a, "b": cluster_id_b,
                                              "new": new_id}))
        return new_id

    def split_clip_out(self, clip_id: int) -> int:
        """Pull ``clip_id`` out of its containing cluster into a singleton."""
        cur = self._ensure_clusters()
        host = next((c for c in cur if clip_id in c.member_clip_ids), None)
        if host is None:
            raise KeyError(f"clip {clip_id} not in any current cluster")
        host.member_clip_ids.remove(clip_id)
        host.size -= 1
        new_id = max((c.id for c in cur), default=-1) + 1
        clip = self.store.get_clip(clip_id)
        cur.append(Cluster(
            id=new_id, parent_class=clip.parent_class,
            exemplar_clip_id=clip_id, member_clip_ids=[clip_id], size=1,
            suggested_labels=self.bank.match(clip.parent_class,
                                             clip.embedding.vector)[:self.cfg.max_suggested_labels],
            intra_distance_mean=0.0,
        ))
        self._cluster_cache = sorted(cur, key=lambda c: c.id)
        self._changes.append(_Change(op="split_clip_out",
                                     payload={"clip_id": clip_id, "new_cluster": new_id}))
        return new_id

    def uncommitted_changes(self) -> list[dict]:
        return [{"op": c.op, **c.payload} for c in self._changes]

    def commit(self, note: Optional[str] = None) -> int:
        sid = self._snapshot(note or f"commit-session-{self.session_id}")
        self._changes.clear()
        return sid

    def rollback(self) -> None:
        """Restore bank state to the entry snapshot."""
        blob = self.store.load_bank_snapshot(self._entry_snapshot_id)
        self.bank._store = pickle.loads(blob)
        self._changes.clear()
        self._cluster_cache = None
