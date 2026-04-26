"""Configuration dataclasses. Every threshold lives here — no magic numbers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    name: str = "handcrafted_v1"
    window_frames_before: int = 8
    window_frames_after: int = 4
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    device: str = "cpu"
    seed: int = 0


@dataclass
class BankConfig:
    split_variance_threshold: float = 0.35
    max_prototypes_per_subclass: int = 4
    online_update_alpha: float = 0.2  # reserved (Welford is parameter-free)


@dataclass
class GateConfig:
    known_distance_threshold: float = 0.25
    ambiguous_distance_threshold: float = 0.45
    min_margin_ratio: float = 1.3
    noise_magnitude_floor: float = 0.05


@dataclass
class StoreConfig:
    db_path: str = "./recognizer.db"
    pose_dtype: str = "float16"
    compress: bool = True


@dataclass
class ReviewConfig:
    min_cluster_size: int = 3
    cluster_selection_epsilon: float = 0.0
    max_suggested_labels: int = 3


@dataclass
class RecognizerConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bank: BankConfig = field(default_factory=BankConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
