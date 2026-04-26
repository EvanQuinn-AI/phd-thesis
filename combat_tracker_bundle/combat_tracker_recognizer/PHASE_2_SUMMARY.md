# Phase 2 summary

Persistence + integration. The recognizer is now driveable end-to-end
from a stream of `AttributedAction` events.

## Built

- `store/schema.sql` — `clips`, `labels`, `prototype_versions`,
  `schema_version`. Indexes on session, parent class, and gate
  decision.
- `store/migrations.py` — versioned, idempotent. v1 = the schema above.
  Re-running on an existing DB is a no-op.
- `store/clipstore.py` — `ClipStore` with the full surface from the
  plan: `store_clip`, `get_clip`, `get_unlabeled`, `get_labeled`,
  `label_clip`, `discard_clip`, `relabel_clip`, `save_bank_snapshot` /
  `load_bank_snapshot` / `list_bank_snapshots`, `export`, `import_`.
  Pose stored as lz4-compressed float16; embeddings stored as raw
  float32. Foreign-key cascade on label deletion.
- `recognizer.py` — `SubclassActionRecognizer` ties encoder + bank +
  gate + store. `observe()` routes through the gate; KNOWN/NOISE
  return without persisting; AMBIGUOUS/UNKNOWN persist a `Clip` and
  return its id for review. `bank_status()` and `rebuild_bank()`
  exposed for ops.
- `integration.py` — `CombatTrackerEventConsumer`. Maintains a per-track
  keypoint ring buffer (`push_track_keypoints` per frame), accepts
  `AttributedAction` events via `observe_action`, queues events whose
  pre-window history is too short, drains queued events on `tick`
  once `window_after` frames have arrived, and `flush()` at end of
  video.

## Tests

`pytest combat_tracker_recognizer/tests/`: **36 passed, 1 xpass**.

New tests:
- `test_store.py` (8 tests): clip round-trip, get_unlabeled exclusion,
  FK cascade, idempotent migrations, export/import, relabel history,
  bank snapshots, import-merge-not-implemented.
- `test_end_to_end.py` (4 tests): unknown→known progression with seeded
  bank, DB survives reconnect, action-queueing for short-history case,
  KNOWN results are not persisted.

## Smoke test

The `test_e2e_unknown_then_known` test in `test_end_to_end.py` is the
Phase 2 smoke test: build the recognizer + consumer, push synthetic
keypoints, fire two `FakeAttribution` events, verify routing.

## Two correctness fixes

Both bugs were caught by the new tests on the first run.

1. **`get_unlabeled` SQL.** The plan's phrasing "no non-discarded row
   in `labels`" treated discarded clips as still-unlabeled. Practical
   intent is the opposite — discard is a triage action that removes a
   clip from the review queue. SQL changed to `id NOT IN (SELECT
   clip_id FROM labels)` (any label kind drops the clip from the
   queue). Recorded in DEVIATIONS.md as a clarification of intent.
2. **Consumer drain dispatches with `force=True`.** Without it, an
   action that fires near the start of the video can never be drained
   because the dispatch path required `>= window_before` frames of
   history before it would commit. After the `window_after` deadline
   passes, dispatching with whatever we have is strictly better than
   dropping the action.

## Known limitations

- `import_(merge=True)` not implemented — raises `NotImplementedError`
  per the plan's "start with replace-only".
- Bank snapshots in this phase live in two places: SQLite
  `prototype_versions` (via `recognizer.rebuild_bank`) and the
  Phase-1 pickle directory (via `bank.versioning`). Phase 3 review
  session will use the SQLite path.
- The `BankSnapshot` dataclass from Phase 1 doesn't surface in Phase 2;
  callers go through `ClipStore.list_bank_snapshots()` which returns
  raw row dicts.

## Ready for Phase 3

- `ClipStore.get_unlabeled` returns the input set for clustering.
- `bank.match` can be queried per-cluster medoid for suggested labels.
- `SubclassActionRecognizer.bank` is mutable from outside (the review
  session will call `bank.add` after labeling).
- `save_bank_snapshot` is the auto-snapshot point on `commit`.
