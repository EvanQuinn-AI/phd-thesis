# Recognizer demo on the sample PvP clip

End-to-end run of `combat_tracker_recognizer` over the same 745-frame
sample clip the tracker uses (`Combat Sports Automation PvP/data/12.mp4`).

## What the demo does

1. **Pass 1** — empty bank. Run YOLO + the parent tracker. For every
   YOLO action detection (cross / hook / kick / high-guard / low-guard)
   call `ActionOwnership.assign` to attribute the action to a track,
   then push the track's pose-window through `CombatTrackerEventConsumer.observe_action`. With no prototypes, every event routes
   `UNKNOWN` and lands in the SQLite store.
2. **Auto-label** — open a `ReviewSession`, list the HDBSCAN clusters
   produced by the unlabeled clips, and label each with a synthetic
   subclass (`punch_01`, `punch_02`, ..., `kick_07`, ...). This stands
   in for human review; the demo doesn't try to assign meaningful names.
3. **Pass 2** — bank seeded. Re-run the same video. Events that match
   a labeled prototype tightly route `KNOWN`; events that match two
   prototypes within margin route `AMBIGUOUS`; the rest stay
   `UNKNOWN`.

Re-generate with:

```bash
python tracking_eval/run_recognizer.py \
    --video "Combat Sports Automation PvP/data/12.mp4" \
    --weights "Combat Sports Automation PvP/models/best.pt"
```

## Headline numbers (`12_summary.json`)

| stage | UNKNOWN | KNOWN | AMBIGUOUS | total |
|---|---:|---:|---:|---:|
| Pass 1 (empty bank) | **344** | 0 | 0 | 344 |
| Pass 2 (after labeling 107 clusters) | 257 | **67** | 20 | 344 |

Events captured per parent class on each pass: punch=38, guard=243,
kick=57. (Guards dominate because YOLO emits a high-guard or low-guard
detection on most frames where a fighter has hands up.)

KNOWN events on pass 2 by parent: `guard=53, kick=9, punch=5`.

## Files

- `12_pass1_results.csv` — every action event observed in pass 1 with its
  gate decision (all UNKNOWN) and the SQLite clip id where it was
  persisted.
- `12_pass2_results.csv` — same for pass 2. The `subclass`,
  `confidence`, and `top_match` columns are populated for KNOWN /
  AMBIGUOUS rows.
- `12_clusters.txt` — one line per HDBSCAN cluster from pass 1, with
  parent class, size, intra-distance mean, and the synthetic label
  assigned.
- `12_summary.json` — aggregate counters, the bank's full subclass
  list after labeling, snapshot count.

## Caveats

- The handcrafted encoder is **random-init** (per the build plan's
  Phase 1 "trained encoder arrives later"). Distances inside a class
  are ~1e-5; between classes they're ~1e-3. `GateConfig` defaults
  (`known=0.25`) would route everything KNOWN; the demo overrides them
  to `known=1e-4 / ambiguous=5e-4 / margin=2.0` so that meaningful
  separation is detectable. See `combat_tracker_recognizer/DEVIATIONS.md`
  D3.
- HDBSCAN over-splits with random embeddings: 107 clusters from 344
  unlabeled clips means almost everything is a singleton. With trained
  weights and the default `min_cluster_size=3` you'd see far fewer,
  more meaningful clusters.
- "Guard" labels are dominated by sustained YOLO detections of a fighter
  holding their hands up. In a real review, those would be filtered out
  before clustering or labeled with a single `guard_active` subclass.
- Labels in the demo are synthetic placeholders (`punch_01`, `kick_07`,
  ...). Replace with `python -m combat_tracker_recognizer review <session>`
  for a real triage session.
