"""Phase B tests: mask-aware PartExtractor."""

import numpy as np

from tracking.base import _chi2_distance
from tracking.features import PartExtractor


def _frame_two_strips(red_left=True):
    """640x480 frame: left half red, right half blue (or swapped)."""
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    left_color = (0, 0, 255) if red_left else (255, 0, 0)
    right_color = (255, 0, 0) if red_left else (0, 0, 255)
    f[:, :320] = left_color
    f[:, 320:] = right_color
    return f


def _landmarks(cx_norm: float):
    return {
        "nose": (cx_norm, 0.25, 0.95),
        "left_shoulder": (cx_norm - 0.05, 0.32, 0.9),
        "right_shoulder": (cx_norm + 0.05, 0.32, 0.9),
        "left_wrist": (cx_norm - 0.08, 0.55, 0.8),
        "right_wrist": (cx_norm + 0.08, 0.55, 0.8),
        "left_hip": (cx_norm - 0.04, 0.6, 0.9),
        "right_hip": (cx_norm + 0.04, 0.6, 0.9),
    }


def test_mask_strips_background_from_torso_histogram():
    """A bbox that straddles the red/blue boundary picks up both colours
    without a mask, but with a mask covering only the red half it picks
    up red alone."""
    frame = _frame_two_strips(red_left=True)
    # bbox straddling the colour boundary, x: 200..440 (overlaps both halves).
    bbox = (200, 100, 440, 380)
    lm = _landmarks(0.5)
    ext = PartExtractor()

    f_no_mask = ext.extract(frame, bbox, lm)
    # Mask covers only the LEFT (red) half within the bbox.
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[:, :320] = 1
    f_masked = ext.extract(frame, bbox, lm, mask=mask)

    # Compare to a clean red-only bbox histogram (ground truth for "red").
    pure_red_bbox = (50, 100, 250, 380)
    f_red = ext.extract(frame, pure_red_bbox, _landmarks(150 / 640))

    common = set(f_red) & set(f_no_mask) & set(f_masked) & {"torso"}
    assert common, "torso missing from one of the histograms"

    d_no_mask = _chi2_distance(f_no_mask["torso"], f_red["torso"])
    d_masked = _chi2_distance(f_masked["torso"], f_red["torso"])
    assert d_masked < d_no_mask, (
        f"mask-aware histogram should be closer to pure-red baseline; "
        f"got d_masked={d_masked:.4f} vs d_no_mask={d_no_mask:.4f}"
    )


def test_mask_aware_extract_is_backwards_compatible():
    """Calling extract without a mask must produce the same output as before."""
    frame = _frame_two_strips()
    bbox = (50, 100, 250, 380)
    lm = _landmarks(150 / 640)
    ext = PartExtractor()
    a = ext.extract(frame, bbox, lm)
    b = ext.extract(frame, bbox, lm, mask=None)
    for region in a:
        assert region in b
        np.testing.assert_allclose(a[region], b[region], atol=1e-6)


def test_region_outside_mask_is_omitted():
    """If a region's bounding box has no mask coverage, the region is dropped."""
    frame = _frame_two_strips()
    bbox = (50, 100, 250, 380)
    lm = _landmarks(150 / 640)
    # Mask covers only the right half — bbox is entirely on the left.
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[:, 320:] = 1
    ext = PartExtractor()
    out = ext.extract(frame, bbox, lm, mask=mask)
    assert out == {}, f"expected empty dict (region fully outside mask), got {out}"


def test_partial_mask_coverage_keeps_region():
    """If at least some pixels in the region are masked-in, the region survives."""
    frame = _frame_two_strips()
    bbox = (200, 100, 440, 380)
    lm = _landmarks(0.5)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[:, :320] = 1  # masks-in only the left portion of the bbox
    ext = PartExtractor()
    out = ext.extract(frame, bbox, lm, mask=mask)
    assert "torso" in out
