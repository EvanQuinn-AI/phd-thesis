# Multi-video qualitative samples

Five runs of the qualitative eval against four different clips, plus a
mask-mode A/B on the flagship one. All produced with:

```bash
python tracking_eval/run_qualitative.py \
    --video <clip> \
    --weights "Combat Sports Automation PvP/models/best.pt" \
    --mode pvp --mask-mode {none|grabcut} [--max-frames N] \
    --out-dir runs/multi/<tag>
```

## Overlay legend

- **Yellow box, "ID 1"** — fighter anchored to the left half of frame.
- **Magenta box, "ID 2"** — fighter anchored to the right half of frame.
- **Red "ID k [CLINCH]"** — clinch detector active for that fighter
  (per-fighter tag, attached to each bbox; replaces the previous top
  banner).
- **Cyan mask outlines + translucent tint** — present only in
  `--mask-mode grabcut`/`yolo_seg` runs. Shows the segmentation the
  tracker is actually using for ReID, clinch detection, and ownership.
- **HAR boxes** (cross / hook / kick / high-guard / low-guard) — drawn
  from raw YOLO action-class detections.
- **Bottom-left HUD** — per-fighter analytics:
  - `thr` throws total
  - `land` landed total (action box must IoU the target's bbox ≥ 0.10
    — proximity alone no longer counts)
  - `hit%` = land / thr
  - `clinch` total seconds spent in clinch
  - `trav` cumulative bbox-centroid travel in pixels
  - `engage avg` mean centre-to-centre distance between fighters
  - `in-range` frames within strike-range threshold (35% of frame
    diagonal)

## Per-clip headline numbers

| clip | frames | duration | ID 1 thr / lnd / hit% | ID 2 thr / lnd / hit% | clinch events | mask mode |
|---|---:|---:|---:|---:|---:|---|
| `12.mp4` | 745 | 24.9s | 75 / 35 / 47% | 30 / 16 / 53% | 9 | none |
| `12.mp4` (mask) | 120 | 4.0s | 20 / 10 / 50% | 5 / 1 / 20% | 1 | grabcut |
| `1.mp4` | 238 | 7.9s | 0 / 0 / 0% | 0 / 0 / 0% | 0 | none |
| `9.mp4` | 558 | 18.3s | 0 / 0 / 0% | 0 / 0 / 0% | 0 | none |
| `15.mp4` | 1028 | 33.7s | 0 / 0 / 0% | 0 / 0 / 0% | 0 | none |

Clips `1.mp4`, `9.mp4`, `15.mp4` are bag-work / single-person scenes.
The PvP weights still emit person + action-class detections on them, but
the 2-slot tracker can't anchor without two simultaneous fighters, so
the analytics legitimately stay at zero. The frames still demonstrate
the overlay (action HAR boxes, the HUD scaffold) on diverse footage.

`12.mp4` is the genuine PvP clip and is the source of the rich
analytics. Hit rates of 47% / 53% are now realistic (the previous
revision over-counted because the target rule picked the closest
non-owner without checking actual contact — fixed by a `landed`
predicate requiring bbox-IoU ≥ 0.10 between action box and target box;
mask-overlap fraction when masks are present).

## Frame samples

Each subdirectory contains 3–5 PNGs at semantically interesting frames
plus the run's `summary.json` with the full analytics dump. Naming:
`frame_NNNN.png` where `NNNN` is the frame index.

- `12/` — open-fight steady-state, mid-clinch (frame 530, "[CLINCH]"
  tag visible on both bboxes), late-game.
- `12_mask/` — same clip with GrabCut masks; cyan outlines and
  translucent tint show the segmentation the tracker is using.
- `1/`, `9/`, `15/` — single-person bag work for diversity; HUD shows
  zeros, action-class boxes still drawn so you can see the YOLO output.

## Mask vs bbox: visible difference

Compare `12/frame_0090.png` (bbox-only) with `12_mask/frame_0090.png`
(grabcut). Both runs use the same tracker, but the mask run additionally
threads instance masks through:
- `PartExtractor` — histograms exclude background pixels
- `ClinchDetector` — mask-IoU instead of bbox-IoU
- `ActionOwnership` — mask-overlap-fraction primary rule

The visible difference in the overlay is the cyan mask outline and tint
on each tracked fighter; the metric difference shows up on tougher
frames where bbox-only misclassifies (heavy bbox overlap with separated
bodies, action centroid in both fighter bboxes). On the open 100-frame
window of 12.mp4 the bbox path was already correct, so the headline
numbers don't move; the test suite (`tests/tracking/test_*_mask.py`)
exercises the win cases on synthetic input.

## Re-generate

```bash
# Bbox-only on each video.
for v in "Combat Sports Automation Tool/data/1.mp4" \
         "Combat Sports Automation Tool/data/9.mp4" \
         "Combat Sports Automation Tool/data/15.mp4" \
         "Combat Sports Automation PvP/data/12.mp4"; do
  base=$(basename "$v" .mp4)
  python tracking_eval/run_qualitative.py --video "$v" \
      --weights "Combat Sports Automation PvP/models/best.pt" \
      --mode pvp --mask-mode none --out-dir "runs/multi/$base"
done

# GrabCut on 12.mp4 (slow — ~1.1 s/frame on CPU; cap with --max-frames).
python tracking_eval/run_qualitative.py \
    --video "Combat Sports Automation PvP/data/12.mp4" \
    --weights "Combat Sports Automation PvP/models/best.pt" \
    --mode pvp --mask-mode grabcut --max-frames 120 \
    --out-dir runs/multi/12_mask
```
