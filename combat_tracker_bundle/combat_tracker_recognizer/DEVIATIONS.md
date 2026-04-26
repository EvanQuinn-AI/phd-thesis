# Deviations from the build plan

Items where the implementation differs from the plan as written. Each
includes: what the plan says, what was done, why, and what (if anything)
needs review.

## D1 — Parent package name

**Plan:** references parent package as `combat_tracker`.

**Done:** parent package in this repo is named `tracking/`. The
recognizer's Phase 2 integration (`CombatTrackerEventConsumer`) will
import from `tracking` rather than `combat_tracker`. Class name kept as
`CombatTrackerEventConsumer` to match the plan's naming.

**Why:** the existing parent package has been `tracking/` since its first
commit. Renaming would touch the Streamlit apps and the existing
qualitative-eval pipeline, which is unrelated work.

**Needs review:** confirm the consumer should use `tracking` as the parent
import or whether to rename `tracking/` → `combat_tracker/` later.

## D2 — `KeypointFormat` parent type

**Plan:** `PoseWindow.keypoint_format: KeypointFormat` mirroring "the
parent package".

**Done:** `tracking/` does not define a `KeypointFormat` enum. The
recognizer defines its own `KeypointFormat` enum
(`combat_tracker_recognizer.types.KeypointFormat`) with
`MEDIAPIPE_13` (the 13-landmark subset that the existing
`pose_analytics.py` exposes) plus `COCO_17` for completeness. If the
parent package later grows a canonical type, the recognizer can re-export
that one and deprecate the local enum.

**Why:** Phase 1 is required to be self-contained; defining the enum
locally avoids creating a parent-package dep that the plan explicitly
forbids in this phase.

**Needs review:** none — the parent has no equivalent type today, so
there's no canonical alternative.

## D3 — Gate-default thresholds

**Plan:** `GateConfig.known_distance_threshold = 0.25`,
`ambiguous_distance_threshold = 0.45`. These are reasonable for cosine
distance on a trained encoder.

**Done:** defaults left at the plan values, BUT with a random-init GRU
the typical intra-class distance is ~1e-5 and inter-class ~2e-3.
Everything passes the 0.25 known gate, so the default config will
classify *anything* as KNOWN until trained weights are available. The
Phase 1 smoke test (`smoke_phase1.py`) overrides the thresholds
explicitly (`known=1e-4`, `ambiguous=5e-4`, `margin=2.0`) to demonstrate
the separation works. The bank tests do not depend on `GateConfig`, so
they are unaffected.

**Why:** changing the plan defaults would silently invalidate any
future trained-weights deployment. Better to keep the defaults aligned
with the encoder we will eventually have, and document the temporary
mismatch.

**Needs review:** confirm that "smoke test overrides defaults" is the
preferred path vs. lowering the defaults until trained weights arrive.

## D4 — `get_unlabeled` interpretation

**Plan:** `get_unlabeled` is "unlabeled means no non-discarded row in
labels". Strict reading: discarded clips ARE unlabeled (since their
only `labels` row has `is_discarded=1`).

**Done:** `get_unlabeled` returns clips with NO row in `labels` at all.
Practically: a discard is a triage decision and removes the clip from
the review queue.

**Why:** the strict reading would resurface discarded clips in every
future review session, defeating the purpose of `discard_clip`. The
plan's broader intent ("review unknowns and either label or discard
them") only works under the practical interpretation.

**Needs review:** none — this is a correctness alignment, not a
deliberate divergence.
