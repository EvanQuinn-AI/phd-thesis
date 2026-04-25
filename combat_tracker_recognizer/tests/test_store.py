"""ClipStore tests."""

from datetime import datetime

import numpy as np
import pytest

from combat_tracker_recognizer.store import ClipStore
from combat_tracker_recognizer.types import (
    Clip,
    Embedding,
    GateDecision,
    KeypointFormat,
    PoseWindow,
)


def _make_clip(parent="punch", session="s1") -> Clip:
    pts = np.random.RandomState(0).uniform(0, 1, (12, 13, 2)).astype(np.float32)
    scs = np.random.RandomState(1).uniform(0.5, 1, (12, 13)).astype(np.float32)
    pw = PoseWindow(points=pts, scores=scs, fps=30.0, frame_start=10, frame_end=21,
                    keypoint_format=KeypointFormat.MEDIAPIPE_13)
    rng = np.random.default_rng(2)
    v = rng.standard_normal(128).astype(np.float32)
    v = v / np.linalg.norm(v)
    emb = Embedding(vector=v, encoder_version="t1", created_at=datetime.utcnow())
    return Clip(parent_class=parent, pose=pw, embedding=emb, session_id=session,
                video_ref="x.mp4", source_track_id=1, similarity_scores={"jab": 0.1})


def test_round_trip_clip(tmp_path):
    store = ClipStore(str(tmp_path / "r.db"))
    c = _make_clip()
    cid = store.store_clip(c, GateDecision.UNKNOWN)
    got = store.get_clip(cid)
    np.testing.assert_allclose(got.pose.points, c.pose.points, atol=1e-2)  # float16 round-trip
    np.testing.assert_allclose(got.pose.scores, c.pose.scores, atol=1e-2)
    np.testing.assert_allclose(got.embedding.vector, c.embedding.vector, atol=1e-6)
    assert got.parent_class == c.parent_class
    assert got.session_id == c.session_id


def test_get_unlabeled_excludes_labeled_and_discarded(tmp_path):
    store = ClipStore(str(tmp_path / "u.db"))
    a = store.store_clip(_make_clip(), GateDecision.UNKNOWN)
    b = store.store_clip(_make_clip(), GateDecision.UNKNOWN)
    c = store.store_clip(_make_clip(), GateDecision.AMBIGUOUS)
    store.label_clip(a, "jab", "punch")
    store.discard_clip(b, note="noise")
    unlabeled = store.get_unlabeled()
    assert {clip.id for clip in unlabeled} == {c}


def test_foreign_key_cascade(tmp_path):
    store = ClipStore(str(tmp_path / "fk.db"))
    cid = store.store_clip(_make_clip(), GateDecision.UNKNOWN)
    store.label_clip(cid, "jab", "punch")
    store._conn.execute("DELETE FROM clips WHERE id = ?", (cid,))
    store._conn.commit()
    rows = store._conn.execute("SELECT * FROM labels WHERE clip_id = ?", (cid,)).fetchall()
    assert rows == []


def test_migrations_idempotent(tmp_path):
    db = str(tmp_path / "m.db")
    s1 = ClipStore(db); s1.close()
    s2 = ClipStore(db)  # should re-run migrations cleanly
    rows = s2._conn.execute("SELECT version FROM schema_version").fetchall()
    assert rows == [(1,)]
    s2.close()


def test_export_import_preserves_data(tmp_path):
    src_path = tmp_path / "src.db"
    dst_path = tmp_path / "dst.db"
    src = ClipStore(str(src_path))
    cid = src.store_clip(_make_clip(), GateDecision.UNKNOWN)
    src.label_clip(cid, "jab", "punch")
    src.export(str(dst_path))
    src.close()

    target = ClipStore(str(tmp_path / "target.db"))
    target.import_(str(dst_path))
    labeled = target.get_labeled()
    assert len(labeled) == 1 and labeled[0][1] == "jab"
    target.close()


def test_relabel_inserts_new_row_does_not_delete(tmp_path):
    store = ClipStore(str(tmp_path / "rl.db"))
    cid = store.store_clip(_make_clip(), GateDecision.UNKNOWN)
    store.label_clip(cid, "jab", "punch")
    store.relabel_clip(cid, "lead_jab", "punch")
    rows = store._conn.execute("SELECT subclass FROM labels WHERE clip_id = ? ORDER BY id",
                               (cid,)).fetchall()
    assert [r[0] for r in rows] == ["jab", "lead_jab"]
    # get_labeled returns the latest.
    labeled = store.get_labeled()
    assert labeled[0][1] == "lead_jab"


def test_save_and_list_bank_snapshots(tmp_path):
    store = ClipStore(str(tmp_path / "sn.db"))
    sid = store.save_bank_snapshot(b"\x00\x01\x02", "v1", note="initial")
    listed = store.list_bank_snapshots()
    assert listed[0]["id"] == sid
    assert listed[0]["note"] == "initial"
    assert store.load_bank_snapshot(sid) == b"\x00\x01\x02"


def test_import_merge_not_implemented(tmp_path):
    store = ClipStore(str(tmp_path / "m.db"))
    other = tmp_path / "other.db"
    other.write_bytes(b"")
    with pytest.raises(NotImplementedError):
        store.import_(str(other), merge=True)
