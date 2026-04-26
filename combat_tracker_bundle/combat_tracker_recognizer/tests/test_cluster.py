"""Clustering tests."""

from datetime import datetime

import numpy as np

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.config import ReviewConfig
from combat_tracker_recognizer.review.cluster import cluster_unknowns
from combat_tracker_recognizer.types import (
    Clip,
    Embedding,
    KeypointFormat,
    PoseWindow,
)


def _empty_pose():
    return PoseWindow(
        points=np.zeros((4, 13, 2), dtype=np.float32),
        scores=np.ones((4, 13), dtype=np.float32),
        fps=30.0, frame_start=0, frame_end=3,
        keypoint_format=KeypointFormat.MEDIAPIPE_13,
    )


def _make_clip(clip_id: int, parent: str, vec: np.ndarray) -> Clip:
    emb = Embedding(vector=vec.astype(np.float32),
                    encoder_version="t", created_at=datetime.utcnow())
    c = Clip(parent_class=parent, pose=_empty_pose(), embedding=emb,
             session_id="s")
    c.id = clip_id
    return c


def test_three_clusters_recovered():
    rng = np.random.default_rng(0)
    centers = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    clips = []
    cid = 0
    for ci, c in enumerate(centers):
        for _ in range(8):
            v = c + 0.05 * rng.standard_normal(4).astype(np.float32)
            v = v / np.linalg.norm(v)
            clips.append(_make_clip(cid, "punch", v))
            cid += 1
    cfg = ReviewConfig(min_cluster_size=4)
    clusters = cluster_unknowns(clips, "punch", cfg)
    assert sum(c.size for c in clusters) == len(clips)
    assert len([c for c in clusters if c.size >= 4]) >= 3


def test_noise_points_become_singletons():
    """The two far outliers must each end up as their own cluster.
    HDBSCAN may split the tight inlier set further; we only require
    that the outliers don't join an inlier cluster."""
    rng = np.random.default_rng(1)
    clips = []
    centre = np.array([1, 0, 0, 0], dtype=np.float32)
    inlier_ids = []
    for i in range(6):
        v = centre + 0.02 * rng.standard_normal(4).astype(np.float32)
        v = v / np.linalg.norm(v)
        clips.append(_make_clip(i, "punch", v))
        inlier_ids.append(i)
    outlier_ids = []
    for i, far in enumerate([np.array([0, 0, 0, 1.0], dtype=np.float32),
                             np.array([0, -1, 0, 0], dtype=np.float32)]):
        clips.append(_make_clip(6 + i, "punch", far / np.linalg.norm(far)))
        outlier_ids.append(6 + i)
    cfg = ReviewConfig(min_cluster_size=4)
    clusters = cluster_unknowns(clips, "punch", cfg)
    # Each outlier must live in a singleton or be alone with the other outlier.
    for oid in outlier_ids:
        host = next(c for c in clusters if oid in c.member_clip_ids)
        assert all(m in outlier_ids or m == oid for m in host.member_clip_ids), \
            f"outlier {oid} ended up with inliers: {host.member_clip_ids}"


def test_medoid_is_in_member_set():
    rng = np.random.default_rng(2)
    clips = []
    centre = np.array([1, 0, 0], dtype=np.float32)
    for i in range(10):
        v = centre + 0.01 * rng.standard_normal(3).astype(np.float32)
        v = v / np.linalg.norm(v)
        clips.append(_make_clip(i, "punch", v))
    cfg = ReviewConfig(min_cluster_size=5)
    clusters = cluster_unknowns(clips, "punch", cfg)
    for c in clusters:
        if c.size > 1:
            assert c.exemplar_clip_id in c.member_clip_ids


def test_suggested_labels_come_from_bank_not_session():
    bank = PrototypeBank()
    rng = np.random.default_rng(3)
    centre = np.array([1, 0, 0, 0], dtype=np.float32)
    centre = centre / np.linalg.norm(centre)
    bank.add("punch", "jab", centre)

    clips = [_make_clip(i, "punch",
                        (centre + 0.01 * rng.standard_normal(4).astype(np.float32))
                        / np.linalg.norm(centre + 0.01 * rng.standard_normal(4)))
             for i in range(6)]
    cfg = ReviewConfig(min_cluster_size=4)
    clusters = cluster_unknowns(clips, "punch", cfg, bank=bank)
    assert clusters
    assert clusters[0].suggested_labels
    assert clusters[0].suggested_labels[0][0] == "jab"


def test_filter_by_parent():
    clips = [_make_clip(0, "punch", np.array([1, 0, 0], dtype=np.float32)),
             _make_clip(1, "kick", np.array([0, 1, 0], dtype=np.float32))]
    cfg = ReviewConfig(min_cluster_size=2)
    clusters = cluster_unknowns(clips, "punch", cfg)
    for c in clusters:
        assert c.parent_class == "punch"
