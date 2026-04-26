"""Identity tracking package for the Combat Sports thesis pipeline.

PvE: single person + bag, with impact attribution.
PvP: two persons, pose-indexed part histograms, clinch-aware ownership.
Vocabulary kept compatible with existing thesis code: track_id, action_owner.
"""

from tracking.base import Track, FeatureBank
from tracking.features import PartExtractor
from tracking.pve import PvETracker, ImpactAttributor
from tracking.pvp import PvPTracker
from tracking.anchoring import IdentityAnchor
from tracking.occlusion import ClinchDetector
from tracking.ownership import ActionOwnership

__all__ = [
    "Track",
    "FeatureBank",
    "PartExtractor",
    "PvETracker",
    "ImpactAttributor",
    "PvPTracker",
    "IdentityAnchor",
    "ClinchDetector",
    "ActionOwnership",
]
