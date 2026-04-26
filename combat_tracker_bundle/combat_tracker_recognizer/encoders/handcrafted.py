"""Hand-crafted-features + GRU encoder.

Per-frame features (order is contractual; bumping any of these requires
incrementing ``HandcraftedEncoder.version`` and rebuilding the bank):

    1. 8 joint angles  (shoulders, elbows, hips, knees)
    2. 8 velocities    (wrists x/y, ankles x/y)
    3. 1 hip rotation proxy (hip-line vs shoulder-line angle)
    4. 1 stance width   (ankle distance / shoulder width)
    5. 1 weight shift   (hip centroid x relative to ankle midpoint x)
    6. 3 stance one-hot (orthodox / southpaw / square)
    7. K visibility mask (one per keypoint, K=13 for MEDIAPIPE_13)

Total per-frame dim = 22 + K. For MEDIAPIPE_13 → 35.

Stack into (T, F), feed into a 2-layer GRU, mean-pool over time, project
to ``embedding_dim`` via a Linear layer, L2-normalise.

Random init is fine for Phase 1 — the bank still works when the encoder
is approximately distance-preserving on similar poses, which a random
GRU mostly is. Trained weights load via ``load_weights(path)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from combat_tracker_recognizer.config import EncoderConfig
from combat_tracker_recognizer.types import (
    KeypointFormat,
    PoseWindow,
    num_keypoints,
)


_MP13_INDEX = {
    "nose": 0,
    "left_shoulder": 1, "right_shoulder": 2,
    "left_elbow": 3, "right_elbow": 4,
    "left_wrist": 5, "right_wrist": 6,
    "left_hip": 7, "right_hip": 8,
    "left_knee": 9, "right_knee": 10,
    "left_ankle": 11, "right_ankle": 12,
}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex b given a-b-c, in radians, vectorised over the leading axis."""
    ba = a - b
    bc = c - b
    cos = (ba * bc).sum(axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _line_angle(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    v = p2 - p1
    return np.arctan2(v[..., 1], v[..., 0])


def _features_per_frame(
    points: np.ndarray,
    scores: np.ndarray,
    visibility_thresh: float = 0.3,
) -> np.ndarray:
    """Compute the per-frame feature matrix (T, F).

    Missing keypoints (score < threshold) get zeroed in their feature
    contributions; the visibility mask preserves the information.
    """
    T, K, _ = points.shape
    idx = _MP13_INDEX
    valid = (scores >= visibility_thresh).astype(np.float32)

    def kp(name: str) -> np.ndarray:
        i = idx[name]
        v = valid[:, i:i + 1]
        return points[:, i] * v

    def kpv(name: str) -> np.ndarray:
        return valid[:, idx[name]]

    ls, rs = kp("left_shoulder"), kp("right_shoulder")
    le, re = kp("left_elbow"), kp("right_elbow")
    lw, rw = kp("left_wrist"), kp("right_wrist")
    lh, rh = kp("left_hip"), kp("right_hip")
    lk, rk = kp("left_knee"), kp("right_knee")
    la, ra = kp("left_ankle"), kp("right_ankle")

    # 1) 8 joint angles
    ang_l_elbow = _angle(ls, le, lw)
    ang_r_elbow = _angle(rs, re, rw)
    ang_l_shoulder = _angle(le, ls, lh)
    ang_r_shoulder = _angle(re, rs, rh)
    ang_l_hip = _angle(ls, lh, lk)
    ang_r_hip = _angle(rs, rh, rk)
    ang_l_knee = _angle(lh, lk, la)
    ang_r_knee = _angle(rh, rk, ra)

    # 2) 8 velocities (forward differences; first frame velocities = 0)
    def vel(arr: np.ndarray) -> np.ndarray:
        v = np.zeros_like(arr)
        v[1:] = arr[1:] - arr[:-1]
        return v

    v_lw = vel(lw)
    v_rw = vel(rw)
    v_la = vel(la)
    v_ra = vel(ra)

    # 3) hip rotation proxy
    hip_angle = _line_angle(lh, rh)
    sh_angle = _line_angle(ls, rs)
    hip_rotation = hip_angle - sh_angle

    # 4) stance width / shoulder width
    ankle_dist = np.linalg.norm(la - ra, axis=-1)
    shoulder_dist = np.linalg.norm(ls - rs, axis=-1) + 1e-6
    stance_width = ankle_dist / shoulder_dist

    # 5) weight shift
    hip_cx = (lh[:, 0] + rh[:, 0]) / 2.0
    ankle_cx = (la[:, 0] + ra[:, 0]) / 2.0
    weight_shift = (hip_cx - ankle_cx) / (shoulder_dist + 1e-6)

    # 6) stance one-hot
    # orthodox = left foot forward (left ankle x < right ankle x for facing-right fighters,
    # but we use a stance-symmetric proxy: smaller-x foot forward).
    forward_l = (la[:, 0] < ra[:, 0]).astype(np.float32)
    backward_l = (la[:, 0] > ra[:, 0]).astype(np.float32)
    square = (np.abs(la[:, 0] - ra[:, 0]) < 0.04).astype(np.float32)
    # Resolve overlap between forward_l and square by giving square priority.
    forward_l = forward_l * (1 - square)
    backward_l = backward_l * (1 - square)

    feats = np.stack([
        ang_l_elbow, ang_r_elbow, ang_l_shoulder, ang_r_shoulder,
        ang_l_hip, ang_r_hip, ang_l_knee, ang_r_knee,
        v_lw[:, 0], v_lw[:, 1], v_rw[:, 0], v_rw[:, 1],
        v_la[:, 0], v_la[:, 1], v_ra[:, 0], v_ra[:, 1],
        hip_rotation, stance_width, weight_shift,
        forward_l, backward_l, square,
    ], axis=1).astype(np.float32)

    # 7) visibility mask (K)
    visibility = valid.astype(np.float32)

    out = np.concatenate([feats, visibility], axis=1)
    # NaN/Inf scrub: any missing-keypoint angles can be NaN.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


class HandcraftedEncoder(nn.Module):
    """Public attributes: ``version``, ``embedding_dim``."""

    version = "handcrafted_v1"

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        torch.manual_seed(config.seed)

        # Feature dim = 22 (engineered) + K (visibility). Computed at first encode().
        self._feature_dim: Optional[int] = None
        self._gru: Optional[nn.GRU] = None
        self._proj: Optional[nn.Linear] = None
        self._device = torch.device(config.device)

    def _ensure_built(self, feature_dim: int) -> None:
        if self._gru is not None:
            return
        self._feature_dim = feature_dim
        # Reseed at build-time so encoder construction order (which is
        # eager) doesn't shift the random state seen by lazy module
        # construction. Two encoders with the same config must produce
        # identical weights regardless of when each first encodes.
        gen_state = torch.get_rng_state()
        torch.manual_seed(self.config.seed)
        try:
            self._gru = nn.GRU(
                input_size=feature_dim,
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                batch_first=True,
            ).to(self._device)
            self._proj = nn.Linear(self.config.hidden_dim, self.embedding_dim).to(self._device)
        finally:
            torch.set_rng_state(gen_state)
        self._gru.eval()
        self._proj.eval()

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        # Build modules with the saved feature dim before loading.
        if "feature_dim" in state:
            self._ensure_built(int(state["feature_dim"]))
        if self._gru is None:
            raise RuntimeError("encoder must be built before load_weights; encode once first")
        self._gru.load_state_dict(state["gru"])
        self._proj.load_state_dict(state["proj"])

    @torch.no_grad()
    def encode(self, window: PoseWindow) -> np.ndarray:
        feats = _features_per_frame(window.points.astype(np.float32),
                                    window.scores.astype(np.float32))
        self._ensure_built(feats.shape[1])
        x = torch.from_numpy(feats).unsqueeze(0).to(self._device)
        out, _ = self._gru(x)
        pooled = out.mean(dim=1)
        emb = self._proj(pooled)
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, windows: list[PoseWindow]) -> np.ndarray:
        if not windows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        # Variable T per window; build one batch by padding to max T.
        feats_list = [_features_per_frame(w.points.astype(np.float32),
                                          w.scores.astype(np.float32)) for w in windows]
        feature_dim = feats_list[0].shape[1]
        self._ensure_built(feature_dim)
        T_max = max(f.shape[0] for f in feats_list)
        batch = np.zeros((len(feats_list), T_max, feature_dim), dtype=np.float32)
        lengths = []
        for i, f in enumerate(feats_list):
            batch[i, :f.shape[0]] = f
            lengths.append(f.shape[0])
        x = torch.from_numpy(batch).to(self._device)
        out, _ = self._gru(x)
        # Mean-pool with length mask.
        mask = torch.zeros(out.shape[:2], device=self._device)
        for i, L in enumerate(lengths):
            mask[i, :L] = 1.0
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        emb = self._proj(pooled)
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().numpy().astype(np.float32)
