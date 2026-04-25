"""Phase 1 tests: PartExtractor + FeatureBank."""

import numpy as np

from tracking.base import FeatureBank
from tracking.features import PartExtractor


def test_part_extractor_distinguishes_two_rectangles(two_rect_frame, two_rect_bboxes, fake_landmarks_two_people):
    ext = PartExtractor()
    f1 = ext.extract(two_rect_frame, two_rect_bboxes[0], fake_landmarks_two_people[0])
    f2 = ext.extract(two_rect_frame, two_rect_bboxes[1], fake_landmarks_two_people[1])

    common = set(f1) & set(f2)
    assert "torso" in common, f"expected torso region, got {common}"

    # Different colours -> non-trivial chi^2 distance on torso.
    from tracking.base import _chi2_distance
    d = _chi2_distance(f1["torso"], f2["torso"])
    assert d > 0.1, f"torso histograms too similar: {d}"


def test_feature_bank_match_under_noise(two_rect_frame, two_rect_bboxes, fake_landmarks_two_people):
    ext = PartExtractor()
    bank_a = FeatureBank()
    bank_b = FeatureBank()

    f_a = ext.extract(two_rect_frame, two_rect_bboxes[0], fake_landmarks_two_people[0])
    f_b = ext.extract(two_rect_frame, two_rect_bboxes[1], fake_landmarks_two_people[1])

    bank_a.add_features(f_a, confidence=0.9)
    bank_b.add_features(f_b, confidence=0.9)

    # Add 10% gaussian noise to f_a and verify it still scores closer to bank_a.
    rng = np.random.default_rng(0)
    noisy_a = {k: np.clip(v + 0.1 * rng.standard_normal(v.shape).astype(v.dtype), 0, None) for k, v in f_a.items()}

    score_to_a = bank_a.score(noisy_a)
    score_to_b = bank_b.score(noisy_a)
    assert score_to_a < score_to_b, f"noisy_a should match bank_a: {score_to_a} vs {score_to_b}"


def test_feature_bank_capacity_evicts_lowest_confidence():
    bank = FeatureBank(capacity=3)
    rng = np.random.default_rng(1)
    for i, conf in enumerate([0.9, 0.5, 0.6, 0.95, 0.4]):
        bank.add("torso", rng.random(256, dtype=np.float32), conf)
    bucket = bank._store["torso"]
    assert len(bucket) == 3
    confs = sorted([c for c, _ in bucket], reverse=True)
    assert confs == [0.95, 0.9, 0.6], confs


def test_part_extractor_no_landmarks_falls_back_to_bbox(two_rect_frame, two_rect_bboxes):
    ext = PartExtractor()
    out = ext.extract(two_rect_frame, two_rect_bboxes[0], landmarks=None)
    assert "torso" in out


def test_part_extractor_skips_low_visibility(two_rect_frame, two_rect_bboxes):
    ext = PartExtractor()
    landmarks = {
        "nose": (0.234, 0.25, 0.1),
        "left_wrist": (0.1, 0.55, 0.9),
    }
    out = ext.extract(two_rect_frame, two_rect_bboxes[0], landmarks)
    assert "head" not in out
    assert "gloves_L" in out
