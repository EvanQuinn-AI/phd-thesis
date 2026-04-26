# Phase 1 summary

Self-contained foundation: types, config, encoder, bank, novelty gate.
No dependency on the parent `tracking/` package.

## Built

- `types.py` — `PoseWindow`, `Embedding`, `Clip`, `GateDecision`,
  `SubclassResult`, `Cluster`, `ReviewDecision`, `KeypointFormat`,
  `MEDIAPIPE_13_NAMES`. Frozen dataclasses where mutation is never
  needed; constructors validate shape/dtype.
- `config.py` — `RecognizerConfig` with five nested sub-configs
  (`EncoderConfig`, `BankConfig`, `GateConfig`, `StoreConfig`,
  `ReviewConfig`). Every threshold is a named field.
- `encoders/` — `Encoder` Protocol, named-encoder registry,
  `HandcraftedEncoder` (22 engineered features per frame + K
  visibility flags → 2-layer GRU → linear projection → L2-normalised
  embedding). Built lazily on first `encode()` with reseeded RNG so
  encoder construction order doesn't perturb determinism.
- `bank/` — `Prototype` (Welford-online mean and variance),
  `PrototypeBank` (parent-scoped match, add, split via k-means k=2,
  merge via Welford parallel formula, rename, remove,
  `rebuild_from_clips`), `versioning.py` (pickle-backed snapshots,
  `save_snapshot` / `load_snapshot` / `list_snapshots` / `rollback`).
- `gate/` — `NoveltyGate` with the four-way routing and the
  margin-ratio rule.

## Tests

`pytest combat_tracker_recognizer/tests/`: **24 passed, 1 xpass**.

The xpassing test is the "encoder is invariant under left-right mirror"
case. It is marked xfail because random-init GRU weights cannot
guarantee invariance; the random init in this run happens to give a
positive cosine. The mark stays — it should fail again as soon as the
encoder is retrained, and that's the intended signal.

## Smoke test

`python -m combat_tracker_recognizer.smoke_phase1` runs in <2 s and
demonstrates the contract: jab seeds the bank, subsequent jabs route
KNOWN, hooks route UNKNOWN.

## Known limitations / deviations

See `DEVIATIONS.md`.

- **D1 — parent name.** The plan refers to `combat_tracker`; the parent
  package in this repo is `tracking/`. Phase 1 is self-contained so
  this is invisible here, but the Phase 2 integration shim will need a
  decision on whether to rename the parent.
- **D2 — KeypointFormat.** Local enum used because the parent has no
  equivalent type yet.
- **D3 — gate defaults.** The plan's default thresholds (`known=0.25`,
  `ambiguous=0.45`) are appropriate for cosine distance on a trained
  encoder. Random-init distances are 100–10,000× tighter, so the
  smoke test overrides the gate config explicitly to show separation;
  defaults left untouched so trained weights drop in cleanly.

## Ready for Phase 2

- `Clip` already carries the fields the SQLite schema expects.
- `Prototype` has `encoder_version` so the rebuild path can detect
  staleness.
- `BankSnapshot` returns are `dataclass`-shaped so the SQLite
  `prototype_versions` table can persist them with one
  `dataclasses.asdict()` call when the time comes.
- `Encoder` Protocol means swapping in a trained encoder later won't
  break Phase 2 wiring.
