# Multi-video qualitative samples

Five runs of the qualitative eval against four different clips, plus a
mask-mode A/B on the flagship one.

```bash
python tracking_eval/run_qualitative.py \
    --video <clip> \
    --weights "Combat Sports Automation PvP/models/best.pt" \
    --mode pvp --mask-mode {none|grabcut} [--max-frames N] \
    --out-dir runs/multi/<tag>
```

## Overlay legend

- **Yellow box, "ID 1"** — left-anchored fighter (or the lone fighter
  in single-person bag-work clips).
- **Magenta box, "ID 2"** — right-anchored fighter (PvP clips only).
- **Red "ID k [CLINCH]"** — per-fighter clinch tag (replaces the old
  top banner).
- **Yellow / magenta skeleton** — per-instance YOLOv8-pose keypoints
  (`models/yolov8n-pose.pt`), drawn in the slot's colour.
- **Cyan mask outline + translucent tint** — instance mask used by
  `PartExtractor` / `ClinchDetector` / `ActionOwnership`. Bag bbox and
  the other fighter's bbox are excluded from grabcut so the mask wraps
  only the fighter.
- **Green box "bag"** — bag detection (PvE-style clips).
- **HAR boxes** drawn ON TOP of all other layers: pink = cross, orange =
  hook, cyan = kick, light green = high-guard, yellow = low-guard.
- **HUD (bottom-left)** — per-fighter throws / landed / hit% / clinch
  seconds / travel-px, plus engagement distance + in-range frame count.

## Per-clip headline numbers

| clip | mode | frames | ID 1 thr / lnd / hit% | ID 2 thr / lnd / hit% | clinch evts | mask |
|---|---|---:|---:|---:|---:|---|
| `12.mp4` | PvP | 745 | 70 / 36 / 51% | 36 / 16 / 44% | 9 | none |
| `12.mp4` | PvP | 120 | 17 / 10 / 59% | 9 / 1 / 11% | 1 | grabcut |
| `1.mp4`  | bag-work | 238 | 94 / 0 / 0% | — | 0 | none |
| `9.mp4`  | bag-work | 558 | 298 / 22 / 7% | — | 0 | none |
| `15.mp4` | bag-work | 1028 | 424 / 18 / 4% | — | 0 | none |

Bag-work clips populate slot 1 only (anchor's single-fighter mode);
slot 2 stays empty. Throws are counted from any cross / hook / kick
detection attributed to the fighter; landed requires the action bbox to
IoU the bag (or the opponent in PvP) by at least 0.10. Hit rates of 4–7%
on the bag-work clips reflect that not every YOLO-detected strike-class
box geometrically intersects the bag — many are mid-air punches that
don't (yet) reach. With trained YOLOv26-pose + seg, mask-overlap-based
hit detection should tighten these numbers.

## What changed since the previous push

- **Per-instance YOLOv8-pose** now drives the pose layer
  (`models/yolov8n-pose.pt`, COCO-17 → MediaPipe-13 mapping). Falls
  back to MediaPipe if the YOLO model isn't available. Skeletons are
  drawn on the overlay.
- **Tightened GrabCut masks**: bag bboxes and other-fighter bboxes are
  forced to definite-background; the output is constrained to the
  input bbox so masks never bleed onto walls or bags.
- **HAR boxes always on top** — they were getting buried under masks
  + bbox borders before. New draw order: masks → bag → dropped
  detections → fighter bboxes → skeletons → **HAR action boxes** → HUD.
- **Single-fighter analytics** — `IdentityAnchor.finalize` now
  populates slot 1 for clips with one consistent fighter. Throws,
  travel, wrist speed, etc. all populate; the bag is the target for
  landed counts via `ActionOwnership.assign(..., bag_box=...)`.
- **Realistic hit rates** — the previous "everything within frame is
  a hit" bug was fixed in the prior commit (action bbox must IoU the
  target bbox ≥ 0.10), so hit% on 12.mp4 is now 51% / 44%.

## Frame samples

- `12/` — flagship PvP. Frames 40, 90, 200, 530 (mid-clinch with
  "[CLINCH]" tags), 700.
- `12_mask/` — same clip, grabcut masks visible (cyan outlines wrap
  only the fighters).
- `1/`, `9/`, `15/` — bag work. ID 1 box + skeleton on the boxer; bag
  bbox visible; HUD shows real throw / landed counts.

## Re-generate

```bash
for v in "Combat Sports Automation Tool/data/1.mp4" \
         "Combat Sports Automation Tool/data/9.mp4" \
         "Combat Sports Automation Tool/data/15.mp4" \
         "Combat Sports Automation PvP/data/12.mp4"; do
  base=$(basename "$v" .mp4)
  python tracking_eval/run_qualitative.py --video "$v" \
      --weights "Combat Sports Automation PvP/models/best.pt" \
      --mode pvp --mask-mode none --out-dir "runs/multi/$base"
done

python tracking_eval/run_qualitative.py \
    --video "Combat Sports Automation PvP/data/12.mp4" \
    --weights "Combat Sports Automation PvP/models/best.pt" \
    --mode pvp --mask-mode grabcut --max-frames 120 \
    --out-dir runs/multi/12_mask
```

## Caveats

- `models/yolov8n-pose.pt` is a generic COCO-17 pose model from
  Ultralytics. A YOLOv26 boxing-specific pose model would lift accuracy
  on tight clinches / unusual stances; the integration path here works
  with any v8/v11/v26 pose checkpoint.
- GrabCut takes ~535 ms/bbox so the mask demo bounds at 120 frames.
  A real seg-model run would push that to <50 ms/frame.
