"""Adapter helpers for slotting tracking/ into the existing Streamlit apps.

The PvP app maintains a ``tracked`` dict with extra bookkeeping fields
(``action_counts``, ``last_hit_frame``) that the legacy
``update_two_person_ids`` function doesn't touch. This adapter calls the
new ``PvPTracker`` and merges its outputs into the existing dict
in-place, leaving bookkeeping intact.

Activation: set the env var ``USE_TRACKING_V2=1`` in the shell where
the Streamlit app is launched. Defaults to legacy behaviour.
"""

from __future__ import annotations

import os
from typing import Optional

from tracking.pvp import PvPTracker


def is_v2_enabled() -> bool:
    return os.environ.get("USE_TRACKING_V2", "0") == "1"


def update_two_person_ids_v2(
    frame,
    person_boxes,
    tracked: dict,
    tracker: PvPTracker,
    landmarks_per_person: Optional[list] = None,
) -> dict:
    """Drop-in replacement for the legacy ``update_two_person_ids``.

    Only the ``box`` and ``hist`` fields of the existing ``tracked`` dict
    are overwritten; ``action_counts`` and ``last_hit_frame`` are
    preserved so downstream counting logic in the Streamlit app keeps
    working untouched.
    """
    new_state = tracker.update(frame, person_boxes, landmarks_per_person)
    for tid in ("1", "2"):
        if tid not in tracked:
            continue
        tracked[tid]["box"] = new_state[tid]["box"]
        tracked[tid]["hist"] = new_state[tid]["hist"]
    return tracked
