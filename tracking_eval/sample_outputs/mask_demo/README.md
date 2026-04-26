# Mask-aware tracker A/B demo (Phases A–F)

This directory holds the side-by-side outputs of the mask-aware tracker
upgrade described in the plan's "Future Architecture: YOLOv26 Instance
Segmentation × Pose × Bbox" section. Phases A–F are committed on
`claude/boxing-identity-tracking-BuRRr`; this is the empirical proof
they run end-to-end on the real PvP clip.

## What was run

Two passes over the first 100 frames of
`Combat Sports Automation PvP/data/12.mp4` (bounded so the GrabCut path
fits in ~2 min instead of ~15 min on the full clip):

```bash
# A — bbox-only (legacy, the path that's been on this branch since Phase 7).
python tracking_eval/run_qualitative.py \
    --video "Combat Sports Automation PvP/data/12.mp4" \
    --weights "Combat Sports Automation PvP/models/best.pt" \
    --mode pvp --mask-mode none --max-frames 100 \
    --out-dir runs/tracking_v3_bbox

# B — mask-aware path; per-bbox masks synthesised via GrabCut so we can
# demonstrate the path without a trained YOLOv26-seg checkpoint.
python tracking_eval/run_qualitative.py \
    --video "Combat Sports Automation PvP/data/12.mp4" \
    --weights "Combat Sports Automation PvP/models/best.pt" \
    --mode pvp --mask-mode grabcut --max-frames 100 \
    --out-dir runs/tracking_v3_mask
```

## Results on the 100-frame window

| metric | bbox-only (A) | grabcut masks (B) |
|---|---:|---:|
| frames processed | 100 | 100 |
| ID 1 visible | 71 | 71 |
| ID 2 visible | 71 | 71 |
| clinch events | 1 | 1 |
| frames with masks used | 0 | 100 |

The bbox-only path was already strong on this clip's opening 100
frames, so the headline numbers don't move. The point of the demo is
**that the mask path runs end-to-end with no regressions**:

- `frames_with_masks: 100` — every kept person detection got a synthetic
  GrabCut mask, every mask was threaded through `PartExtractor`,
  `ClinchDetector`, and (for action events) `ActionOwnership`.
- `id1_visible / id2_visible` unchanged — the contamination gate and
  Kalman-with-bank update logic survive the mask threading.
- The mask path's value shows up empirically on cases the bbox-only
  path mishandles: heavy bbox overlap with separated bodies (so bbox-IoU
  trips clinch but mask-IoU doesn't), and action attribution where two
  fighter bboxes both contain the action centroid (kinematic + mask
  win the tie). These are the cases the unit tests exercise — see
  `tests/tracking/test_features_mask.py`,
  `tests/tracking/test_occlusion_mask.py`,
  `tests/tracking/test_pvp_mask.py`,
  `tests/tracking/test_ownership_mask.py`.

## Frame samples

Same 4 frame indices, both modes. The mask-mode overlay is identical to
the bbox-mode overlay because the synthesised masks only affect the
internal feature/clinch/ownership logic, not the rendered overlay.
A future overlay tweak could draw the masks themselves; for now the
proof is the JSON summary + the unit test coverage.

- `frames/bbox_030.png` vs `frames/mask_030.png`
- `frames/bbox_050.png` vs `frames/mask_050.png`
- `frames/bbox_075.png` vs `frames/mask_075.png`
- `frames/bbox_095.png` vs `frames/mask_095.png`

## What this isn't

- **Not a trained-model run.** Real YOLOv26-seg weights would replace
  GrabCut and run at the segmentation-head's native speed (~50 ms/frame
  rather than ~1.1 s/frame). They'd also produce per-instance masks
  rather than per-bbox masks, fixing the case where two fighters share
  one bbox.
- **Not the full clip.** GrabCut is the bottleneck at 535 ms/bbox in
  this sandbox; running the full 745 frames takes ~14 min.
  `--max-frames 100` keeps the demo turnaround tractable. The full-clip
  run would be the right artefact for an ablation table once a real
  seg model exists.

## Test status

```
tests/tracking/                  73 passed
combat_tracker_recognizer/tests/  48 passed
total                            121 passed
```
