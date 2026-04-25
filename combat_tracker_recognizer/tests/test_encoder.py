"""Encoder tests."""

import numpy as np
import pytest

from combat_tracker_recognizer.config import EncoderConfig
from combat_tracker_recognizer.encoders import HandcraftedEncoder, get_encoder, list_encoders


def test_handcrafted_is_registered():
    assert "handcrafted_v1" in list_encoders()


def test_encode_is_deterministic_with_fixed_seed(jab_window):
    e1 = HandcraftedEncoder(EncoderConfig(seed=42))
    e2 = HandcraftedEncoder(EncoderConfig(seed=42))
    v1 = e1.encode(jab_window)
    v2 = e2.encode(jab_window)
    np.testing.assert_allclose(v1, v2, rtol=1e-5, atol=1e-6)


def test_encoding_shape_matches_embedding_dim(jab_window):
    e = HandcraftedEncoder(EncoderConfig(embedding_dim=128, seed=1))
    v = e.encode(jab_window)
    assert v.shape == (128,) and v.dtype == np.float32


def test_encoding_is_l2_normalised(jab_window):
    e = HandcraftedEncoder(EncoderConfig(seed=1))
    v = e.encode(jab_window)
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_missing_keypoints_does_not_produce_nan(jab_window):
    bad_scores = np.zeros_like(jab_window.scores)
    pw = type(jab_window)(
        points=jab_window.points, scores=bad_scores,
        fps=jab_window.fps, frame_start=0, frame_end=jab_window.num_frames - 1,
        keypoint_format=jab_window.keypoint_format,
    )
    e = HandcraftedEncoder(EncoderConfig(seed=1))
    v = e.encode(pw)
    assert not np.isnan(v).any()
    assert not np.isinf(v).any()


def test_encode_batch_matches_encode_per_window(jab_window, hook_window):
    e = HandcraftedEncoder(EncoderConfig(seed=7))
    individual = np.stack([e.encode(jab_window), e.encode(hook_window)])
    batched = e.encode_batch([jab_window, hook_window])
    np.testing.assert_allclose(individual, batched, atol=1e-5)


def test_get_encoder_via_registry():
    e = get_encoder("handcrafted_v1", EncoderConfig(seed=1))
    assert e.version == "handcrafted_v1"
    assert e.embedding_dim == 128


@pytest.mark.xfail(reason="random-init encoder; stance invariance only expected with trained weights")
def test_orthodox_and_southpaw_mirror_are_close(jab_window):
    """A jab with the body mirrored left-right should embed close to the
    original. With random GRU weights this is unreliable — flagged xfail."""
    e = HandcraftedEncoder(EncoderConfig(seed=1))
    pts_mirror = jab_window.points.copy()
    pts_mirror[..., 0] = 1.0 - pts_mirror[..., 0]
    mirrored = type(jab_window)(
        points=pts_mirror, scores=jab_window.scores,
        fps=jab_window.fps, frame_start=0, frame_end=jab_window.num_frames - 1,
        keypoint_format=jab_window.keypoint_format,
    )
    v_orig = e.encode(jab_window)
    v_mirror = e.encode(mirrored)
    cos_self = float(v_orig @ v_orig)
    cos_mirror = float(v_orig @ v_mirror)
    assert cos_mirror > 0.5 * cos_self
