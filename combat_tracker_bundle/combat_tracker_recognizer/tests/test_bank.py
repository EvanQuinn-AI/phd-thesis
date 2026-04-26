"""Bank tests."""

import numpy as np
import pytest

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.bank.prototype import Prototype


def _vec(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(8).astype(np.float32)
    return v / np.linalg.norm(v)


def test_add_makes_subclass_matchable():
    bank = PrototypeBank()
    e = _vec(0)
    bank.add("punch", "jab", e)
    matches = bank.match("punch", e)
    assert matches and matches[0][0] == "jab"
    assert matches[0][1] < 1e-3


def test_match_never_returns_cross_parent():
    bank = PrototypeBank()
    e = _vec(0)
    bank.add("punch", "jab", e)
    bank.add("kick", "roundhouse", e)
    matches = bank.match("punch", e)
    assert all(m[0] == "jab" for m in matches)


def test_split_preserves_total_exemplars():
    bank = PrototypeBank()
    for i in range(8):
        bank.add("punch", "jab", _vec(i))
    total_before = sum(len(p.exemplar_embeddings) for p in bank.get("punch", "jab"))
    bank.split("punch", "jab")
    keys_after = bank.all_subclasses("punch")
    total_after = sum(len(p.exemplar_embeddings)
                      for k in keys_after
                      for p in bank.get(*k))
    assert total_after == total_before
    assert len(keys_after) == 2


def test_merge_mean_is_correct_weighted_average():
    bank = PrototypeBank()
    bank.add("punch", "a", _vec(0))
    bank.add("punch", "a", _vec(1))
    bank.add("punch", "a", _vec(2))
    bank.add("punch", "b", _vec(10))
    bank.add("punch", "b", _vec(11))
    proto_a = bank.get("punch", "a")[0]
    proto_b = bank.get("punch", "b")[0]
    n_a, mean_a = proto_a.exemplar_count, proto_a.mean.copy()
    n_b, mean_b = proto_b.exemplar_count, proto_b.mean.copy()
    expected = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    bank.merge("punch", "a", "b", "ab")
    proto_ab = bank.get("punch", "ab")[0]
    np.testing.assert_allclose(proto_ab.mean, expected, atol=1e-5)
    assert proto_ab.exemplar_count == n_a + n_b


def test_rename_preserves_prototypes():
    bank = PrototypeBank()
    bank.add("punch", "jab", _vec(0))
    bank.rename("punch", "jab", "right_jab")
    assert bank.match("punch", _vec(0))[0][0] == "right_jab"


def test_remove_drops_subclass():
    bank = PrototypeBank()
    bank.add("punch", "jab", _vec(0))
    bank.remove("punch", "jab")
    assert bank.match("punch", _vec(0)) == []


def test_welford_variance_matches_numpy():
    p = Prototype.from_first("punch", "jab", _vec(0), encoder_version="t")
    samples = [_vec(0)]
    for i in range(1, 30):
        v = _vec(i)
        p.update(v)
        samples.append(v)
    expected_var = np.var(np.stack(samples), axis=0, ddof=1)
    np.testing.assert_allclose(p.variance(), expected_var, atol=1e-5)


def test_rebuild_from_clips_with_same_encoder_is_equivalent():
    """Stand-in test using a fake encoder that returns the embedding off the clip."""
    from datetime import datetime
    from combat_tracker_recognizer.types import Embedding, PoseWindow, KeypointFormat

    pts = np.zeros((10, 13, 2), dtype=np.float32)
    sc = np.ones((10, 13), dtype=np.float32)
    pw = PoseWindow(points=pts, scores=sc, fps=30.0, frame_start=0, frame_end=9,
                    keypoint_format=KeypointFormat.MEDIAPIPE_13)

    class FakeEncoder:
        version = "fake_v1"
        embedding_dim = 8

        def __init__(self):
            self._counter = 0

        def encode(self, window):
            self._counter += 1
            v = _vec(self._counter)
            return v

        def encode_batch(self, windows):
            return np.stack([self.encode(w) for w in windows])

    class StubLabeledClip:
        def __init__(self, parent, subclass):
            self.parent_class = parent
            self.subclass = subclass
            self.pose = pw
            self.embedding = Embedding(vector=_vec(0), encoder_version="fake_v1",
                                       created_at=datetime.utcnow())

    bank = PrototypeBank()
    encoder = FakeEncoder()
    clips = [StubLabeledClip("punch", "jab"), StubLabeledClip("punch", "cross"),
             StubLabeledClip("kick", "roundhouse")]
    bank.rebuild_from_clips(clips, encoder)
    keys = bank.all_subclasses()
    assert ("punch", "jab") in keys
    assert ("punch", "cross") in keys
    assert ("kick", "roundhouse") in keys
