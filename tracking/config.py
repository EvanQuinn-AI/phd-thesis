"""Tracking configuration. Plain dataclasses to match the repo's no-Hydra style."""

from dataclasses import dataclass, field


@dataclass
class TrackingConfig:
    # Feature extraction
    hist_h_bins: int = 16
    hist_s_bins: int = 16
    landmark_visibility_thresh: float = 0.5
    feature_bank_size: int = 20

    # Kalman / association
    iou_match_threshold: float = 0.5
    max_age_frames: int = 30
    pose_conf_for_bank_update: float = 0.6
    contamination_iou_thresh: float = 0.3

    # Anchoring
    anchor_window_frames: int = 30

    # Clinch
    clinch_iou_thresh: float = 0.6
    clinch_min_frames: int = 5
    disocclusion_uncertain_window: int = 15
    disocclusion_min_match_conf: float = 0.7

    # Region weights for ReID (glove > trunks > head > torso)
    reid_region_weights: dict = field(
        default_factory=lambda: {
            "gloves_L": 1.0,
            "gloves_R": 1.0,
            "trunks": 0.8,
            "head": 0.5,
            "torso": 0.3,
        }
    )


DEFAULT = TrackingConfig()
