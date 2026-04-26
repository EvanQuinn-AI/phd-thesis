"""Instance-mask helpers used by the mask-aware tracker upgrade.

YOLO segmentation models (Ultralytics v8/v11/v26) return per-detection masks
alongside boxes and classes. Detection-only models (e.g. the existing
YOLOv5 ``best.pt``) do not. Code that consumes masks must therefore tolerate
``None`` everywhere — that's the explicit back-compat contract.

This module also provides a GrabCut-based synthetic-mask path. It lets the
qualitative eval demonstrate the mask-aware tracker on the sample clip even
though no segmentation model has been trained on the boxing classes.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary instance masks. Both masks must be the same shape."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return inter / union


def mask_overlap_fraction(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Fraction of ``mask_a``'s pixels that are also in ``mask_b``.

    Used by mask-IoU action ownership: the action mask whose pixels mostly
    sit inside fighter k's mask belongs to fighter k.
    """
    a = mask_a.astype(bool)
    if a.sum() == 0:
        return 0.0
    b = mask_b.astype(bool)
    return float(np.logical_and(a, b).sum() / a.sum())


def bbox_to_mask(bbox: tuple, frame_shape: tuple) -> np.ndarray:
    """Degenerate "mask" that's just the bbox-as-rectangle. Useful for tests."""
    h, w = frame_shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = 1
    return out


def grabcut_mask(frame_bgr: np.ndarray, bbox: tuple,
                 num_iters: int = 3) -> Optional[np.ndarray]:
    """Foreground mask for a single bbox via GrabCut.

    Slow (~50-200 ms per call). Used only in the qualitative eval as a
    stand-in for true instance segmentation, so the mask path can be
    demonstrated without a trained YOLOv26-seg model.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)
    try:
        cv2.grabCut(frame_bgr, mask, rect, bgd, fgd, num_iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return out


def extract_yolo_seg_masks(results, frame_shape: tuple) -> Optional[list[np.ndarray]]:
    """Pull per-detection masks out of an Ultralytics YOLO segmentation result.

    Returns ``None`` if the model is detection-only (no ``masks`` attribute).
    Returns a list of ``(H, W)`` uint8 binary masks aligned with the
    ``xyxy`` detections, otherwise.

    Tolerates the various result shapes across yolov5/v8/v11 by checking
    for the presence of a ``data`` attribute (Ultralytics v8+) or a tensor
    (legacy v5 segmentation export).
    """
    if not hasattr(results, "masks") and not hasattr(results, "xyxy"):
        return None
    masks_attr = getattr(results, "masks", None)
    if masks_attr is None:
        return None

    h, w = frame_shape[:2]

    # Ultralytics v8/v11/v26: results.masks.data is (N, mh, mw) torch tensor
    data = getattr(masks_attr, "data", None)
    if data is not None:
        arr = data.detach().cpu().numpy()
        out = []
        for m in arr:
            resized = cv2.resize(m.astype(np.float32), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
            out.append((resized > 0.5).astype(np.uint8))
        return out

    # Legacy v5-seg export: results.masks is a list of polygons or similar.
    # Fall through and return None so callers go down the no-mask path.
    return None
