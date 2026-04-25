"""combat_tracker_recognizer — subclass action recognition with active learning.

Sibling package to ``tracking/``. Phase 1 is self-contained; Phase 2 wires
into the parent tracker via ``CombatTrackerEventConsumer``.

Public surface kept narrow per the plan:
"""

from combat_tracker_recognizer.config import (
    BankConfig,
    EncoderConfig,
    GateConfig,
    RecognizerConfig,
    ReviewConfig,
    StoreConfig,
)
from combat_tracker_recognizer.types import (
    Cluster,
    Embedding,
    GateDecision,
    PoseWindow,
    SubclassResult,
)

__all__ = [
    "RecognizerConfig",
    "EncoderConfig",
    "BankConfig",
    "GateConfig",
    "StoreConfig",
    "ReviewConfig",
    "PoseWindow",
    "Embedding",
    "GateDecision",
    "SubclassResult",
    "Cluster",
]
