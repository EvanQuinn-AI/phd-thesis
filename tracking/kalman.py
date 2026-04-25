"""Constant-velocity Kalman filter on bbox centre + size.

State: [cx, cy, w, h, vx, vy]. Pure numpy — avoids the filterpy dependency
which fails to build in some environments. Sufficient for 30 fps boxing.
"""

from __future__ import annotations

import numpy as np


def _bbox_to_state(bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
        0.0,
        0.0,
    ], dtype=np.float64)


def _state_to_bbox(state: np.ndarray) -> tuple:
    cx, cy, w, h = state[:4]
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


class BBoxKalman:
    def __init__(self, bbox: tuple, process_noise: float = 1.0, measurement_noise: float = 1.0):
        self.x = _bbox_to_state(bbox)
        self.P = np.eye(6) * 10.0

        # State transition: position += velocity, size constant, velocity constant.
        self.F = np.eye(6)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        # We measure cx, cy, w, h directly.
        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = np.eye(6) * process_noise
        # Velocity noise larger than position noise.
        self.Q[4, 4] = process_noise * 4.0
        self.Q[5, 5] = process_noise * 4.0

        self.R = np.eye(4) * measurement_noise

    def predict(self) -> tuple:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _state_to_bbox(self.x)

    def update(self, bbox: tuple) -> tuple:
        z = np.array([
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
        ], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return _state_to_bbox(self.x)

    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])

    def bbox(self) -> tuple:
        return _state_to_bbox(self.x)


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
