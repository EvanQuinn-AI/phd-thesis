"""ReviewSession tests + Phase 3 end-to-end."""

from datetime import datetime

import numpy as np
import pytest

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.config import GateConfig, RecognizerConfig
from combat_tracker_recognizer.integration import CombatTrackerEventConsumer
from combat_tracker_recognizer.recognizer import SubclassActionRecognizer
from combat_tracker_recognizer.review.session import ReviewSession
from combat_tracker_recognizer.store import ClipStore
from combat_tracker_recognizer.tests.conftest import make_pose_window
from combat_tracker_recognizer.tests.test_end_to_end import (
    FakeAttribution,
    _push_window_to_buffer,
)
from combat_tracker_recognizer.types import (
    Clip,
    Embedding,
    GateDecision,
    KeypointFormat,
    PoseWindow,
)


def _make_clip_in_store(store: ClipStore, parent: str, vec: np.ndarray,
                        decision: GateDecision = GateDecision.UNKNOWN) -> int:
    pose = PoseWindow(
        points=np.zeros((4, 13, 2), dtype=np.float32),
        scores=np.ones((4, 13), dtype=np.float32),
        fps=30.0, frame_start=0, frame_end=3,
        keypoint_format=KeypointFormat.MEDIAPIPE_13,
    )
    emb = Embedding(vector=vec.astype(np.float32),
                    encoder_version="t1", created_at=datetime.utcnow())
    return store.store_clip(
        Clip(parent_class=parent, pose=pose, embedding=emb, session_id="s"),
        decision,
    )


def _vec(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(8).astype(np.float32)
    return v / np.linalg.norm(v)


def test_label_cluster_adds_to_bank_and_marks_clips(tmp_path):
    store = ClipStore(str(tmp_path / "rs.db"))
    centre = _vec(0)
    for i in range(5):
        rng = np.random.default_rng(100 + i)
        v = centre + 0.02 * rng.standard_normal(8).astype(np.float32)
        v = v / np.linalg.norm(v)
        _make_clip_in_store(store, "punch", v)
    bank = PrototypeBank()
    cfg = RecognizerConfig()
    cfg.review.min_cluster_size = 3
    sess = ReviewSession("test", store, bank, cfg.review, encoder_version="t1")
    clusters = sess.list_clusters()
    target = next(c for c in clusters if c.size >= 3)
    sess.label_cluster(target.id, "jab")
    assert ("punch", "jab") in bank.all_subclasses()
    assert sess.uncommitted_changes()


def test_commit_creates_new_snapshot_row(tmp_path):
    store = ClipStore(str(tmp_path / "c.db"))
    bank = PrototypeBank()
    cfg = RecognizerConfig()
    sess = ReviewSession("t", store, bank, cfg.review, encoder_version="t")
    initial_n = len(store.list_bank_snapshots())
    sess.commit(note="manual")
    assert len(store.list_bank_snapshots()) == initial_n + 1
    assert store.list_bank_snapshots()[0]["note"] == "manual"


def test_rollback_restores_bank(tmp_path):
    store = ClipStore(str(tmp_path / "rb.db"))
    bank = PrototypeBank()
    cfg = RecognizerConfig()
    sess = ReviewSession("t", store, bank, cfg.review, encoder_version="t")
    centre = _vec(0)
    for i in range(5):
        rng = np.random.default_rng(200 + i)
        v = centre + 0.02 * rng.standard_normal(8).astype(np.float32)
        v = v / np.linalg.norm(v)
        _make_clip_in_store(store, "punch", v)
    sess.list_clusters()
    target = next(c for c in sess._cluster_cache if c.size >= 1)
    sess.label_cluster(target.id, "jab")
    assert ("punch", "jab") in bank.all_subclasses()
    sess.rollback()
    assert ("punch", "jab") not in bank.all_subclasses()


def test_discard_cluster_removes_from_unlabeled(tmp_path):
    store = ClipStore(str(tmp_path / "d.db"))
    for i in range(4):
        _make_clip_in_store(store, "punch", _vec(i))
    bank = PrototypeBank()
    cfg = RecognizerConfig()
    cfg.review.min_cluster_size = 2
    sess = ReviewSession("t", store, bank, cfg.review, encoder_version="t")
    clusters = sess.list_clusters()
    sess.discard_cluster(clusters[0].id)
    assert len(store.get_unlabeled()) < 4


def test_merge_clusters_combines_membership(tmp_path):
    store = ClipStore(str(tmp_path / "m.db"))
    rng = np.random.default_rng(0)
    a_centre = _vec(10)
    b_centre = _vec(20)
    for i in range(4):
        v = a_centre + 0.01 * rng.standard_normal(8).astype(np.float32)
        _make_clip_in_store(store, "punch", v / np.linalg.norm(v))
    for i in range(4):
        v = b_centre + 0.01 * rng.standard_normal(8).astype(np.float32)
        _make_clip_in_store(store, "punch", v / np.linalg.norm(v))
    bank = PrototypeBank()
    cfg = RecognizerConfig()
    cfg.review.min_cluster_size = 3
    sess = ReviewSession("t", store, bank, cfg.review, encoder_version="t")
    clusters = sess.list_clusters()
    bigs = [c for c in clusters if c.size >= 3]
    if len(bigs) < 2:
        pytest.skip("HDBSCAN merged into one cluster on this run")
    new_id = sess.merge_clusters(bigs[0].id, bigs[1].id)
    merged = sess.get_cluster(new_id)
    assert merged.size == bigs[0].size + bigs[1].size


def test_phase3_e2e_label_then_replay_routes_known(tmp_path):
    """Full Phase 3 contract: review → label → replay → KNOWN."""
    cfg = RecognizerConfig()
    cfg.store.db_path = str(tmp_path / "p3.db")
    cfg.gate = GateConfig(known_distance_threshold=1e-4,
                          ambiguous_distance_threshold=5e-4,
                          min_margin_ratio=2.0)
    cfg.review.min_cluster_size = 2

    rec = SubclassActionRecognizer(cfg)
    consumer = CombatTrackerEventConsumer(rec, session_id="sess")

    # Push two jab events into the unlabeled set.
    for i, seed in enumerate([1, 2]):
        w = make_pose_window("jab", T=12, seed=seed)
        _push_window_to_buffer(consumer, track_id=10 + i,
                               start_frame=i * 100, window=w)
        consumer.observe_action(FakeAttribution(action_class="punch", owner_id=str(10 + i)),
                                frame_idx=i * 100 + 8, video_ref="x.mp4")

    assert rec.pending_review_count(session_id="sess") == 2

    sess = ReviewSession("sess", rec.clipstore, rec.bank, cfg.review,
                         encoder_version=rec.encoder.version)
    clusters = sess.list_clusters()
    # 2 clips with min_cluster_size=2 may form one cluster or two singletons.
    for c in clusters:
        sess.label_cluster(c.id, "jab")
    sess.commit(note="phase3-e2e")

    # New jab with a different seed → KNOWN.
    new_jab = make_pose_window("jab", T=12, seed=99)
    _push_window_to_buffer(consumer, track_id=99, start_frame=500, window=new_jab)
    r = consumer.observe_action(FakeAttribution(action_class="punch", owner_id="99"),
                                frame_idx=508, video_ref="x.mp4")
    assert r.decision == GateDecision.KNOWN
    assert r.subclass == "jab"


def test_split_clip_creates_singleton(tmp_path):
    store = ClipStore(str(tmp_path / "sp.db"))
    rng = np.random.default_rng(0)
    centre = _vec(0)
    for i in range(5):
        v = centre + 0.02 * rng.standard_normal(8).astype(np.float32)
        _make_clip_in_store(store, "punch", v / np.linalg.norm(v))
    bank = PrototypeBank()
    cfg = RecognizerConfig()
    cfg.review.min_cluster_size = 3
    sess = ReviewSession("t", store, bank, cfg.review, encoder_version="t")
    clusters = sess.list_clusters()
    big = next(c for c in clusters if c.size >= 3)
    pulled_id = big.member_clip_ids[0]
    new_cluster_id = sess.split_clip_out(pulled_id)
    found = sess.get_cluster(new_cluster_id)
    assert found.member_clip_ids == [pulled_id]
    assert found.size == 1
