# INTEGRATION.md — instructions for Claude

You are receiving this bundle in a project that already has a YOLOv26
checkpoint (detection or segmentation) trained on combat-sports
classes. Your job is to wire the bundle into that project so the
qualitative-eval scripts run end-to-end against the user's video data.

These instructions are self-contained. Do **not** assume the user has
read the bundle's README or knows the internals.

-----

## Step 0 — Verify the bundle landed correctly

Before changing anything in the user's project, run:

```bash
pytest tests/tracking/
pytest combat_tracker_recognizer/tests/
```

Expected: **79 passed** under `tests/tracking/` and **48 passed +
1 xpass** under `combat_tracker_recognizer/tests/`. If anything fails,
something went wrong during copy — investigate before integrating.

-----

## Step 1 — Identify the YOLOv26 entry points in the user's project

Ask the user (or grep) to find:

1. **Their YOLO loader function**. Most likely in their main Streamlit
   or training script — look for `torch.hub.load`, `YOLO(...)`, or
   similar. Note the function name and which checkpoint path it loads.
2. **Their class-id map**. Look for a `class_names` dict or a YAML
   config (often `data.yaml` for Ultralytics) listing classes in order.

Record:
- Person class id (likely 0 or near it)
- Bag class id (if dataset has a "boxing-bag" class)
- Action class ids (cross / hook / kick / high-guard / low-guard, or
  whatever the dataset uses)

-----

## Step 2 — Wire the YOLO loader into the eval scripts

Open `tracking_eval/run_qualitative.py`. Find `_load_yolo` near the top
of the file:

```python
def _load_yolo(weights: str):
    """Load a YOLOv5 .pt checkpoint."""
    import torch
    _orig_load = torch.load
    def _patched(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    torch.load = _patched
    try:
        try:
            return torch.hub.load("ultralytics/yolov5", "custom", path=weights,
                                  force_reload=False, trust_repo=True)
        except Exception:
            import yolov5
            return yolov5.load(weights)
    finally:
        torch.load = _orig_load
```

Replace the body with whatever loader the user's project uses. Most
YOLOv26 projects on Ultralytics will be:

```python
def _load_yolo(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)
```

If their loader returns a different result shape, also patch
`_yolo_detections(model, frame)` (just below `_load_yolo`) so it
returns a numpy array of `[x1, y1, x2, y2, conf, class_id]` rows. The
Ultralytics standard return is `results[0].boxes.data.cpu().numpy()`
which already matches.

-----

## Step 3 — Update class id maps

Still in `tracking_eval/run_qualitative.py`, find:

```python
_PVE_CLASS_NAMES = { 0: "boxing-bag", 1: "high-guard", ... }
_PVP_CLASS_NAMES = { 0: "boxing-bag", 1: "cross", 2: "high-guard", ... }
```

and the constants:

```python
_PERSON_CLASS_PVE = 4
_BAG_CLASS_PVE = 0
_PERSON_CLASS_PVP = 6
_BAG_CLASS_PVP = 0
```

Update both the dict and the constants to match the user's dataset.
The names on the right of the dict matter — the analytics module
treats names that include `"guard"` as defensive (counted toward
`guards_active_frames`, not throws), and uses `"cross"`, `"hook"`,
`"kick"`, `"punch"`, `"kick-knee"` as strike classes. If their
dataset uses different strings, also update `_TERMINAL_KEYPOINT` in
`tracking/ownership.py` so the kinematic-chain rule maps the right
limb (wrist for punch, ankle for kick) to each class.

-----

## Step 4 — Choose a pose source

The eval runner tries YOLOv8/v11/v26-pose first via
`models/yolov8n-pose.pt`, then falls back to MediaPipe.

If the user has trained a YOLOv26-pose model on the dataset:

```python
def _try_load_yolo_pose(weights: str = "models/yolov8n-pose.pt"):
```

Update the default path to their pose checkpoint. Keep it as a default
arg so the existing `--use-pose-weights` flag (if the user wants to add
one) overrides cleanly.

If they only have detection / seg (no pose), MediaPipe will be used.
That's fine but lower-quality on tight clinches.

-----

## Step 5 — Choose mask mode

If the user's YOLOv26 model is **segmentation** (returns masks per
detection), use:

```bash
python tracking_eval/run_qualitative.py \
    --video <clip> --weights <user's seg .pt> \
    --mode pvp --mask-mode yolo_seg \
    --out-dir runs/qualitative_seg
```

Verify that `extract_yolo_seg_masks` in `tracking/masks.py` correctly
parses their results. The function expects an Ultralytics-style
`results` object with a `masks.data` attribute holding an
`(N, mh, mw)` tensor. If the user's wrapper returns a different shape,
update `extract_yolo_seg_masks` accordingly. **Do NOT silently rewrite
the call sites to use a different shape — the contract is documented in
the docstring; preserve it.**

If the user's model is **detection-only**, use `--mask-mode none`
(the default) or `--mask-mode grabcut` to A/B the mask path with
synthetic foreground masks.

-----

## Step 6 — Run end-to-end on a sample clip

```bash
python tracking_eval/run_qualitative.py \
    --video <user's sample clip> \
    --weights <user's YOLOv26 .pt> \
    --mode pvp --mask-mode <chosen> \
    --out-dir runs/qualitative
```

Outputs:
- `runs/qualitative/<basename>_v2_overlay.mp4` — annotated video.
- `runs/qualitative/<basename>_v2_log.csv` — per-frame state.
- `runs/qualitative/<basename>_v2_summary.json` — aggregate analytics.

In the JSON, look at:
- `analytics.fighters["1"].throws_total` and `landed_total`
- `analytics.fighters["2"].throws_total` and `landed_total`
- `analytics.fighters["1"].shots_received_total` should mirror
  `analytics.fighters["2"].landed_total` (within rounding)
- `clinch_events` — should match what you'd intuitively count

If `throws_total == 0` for both fighters, the class-id maps are
probably wrong. If `shots_received` doesn't mirror `landed`, the
ownership rule may need tuning (open `tracking/ownership.py`).

-----

## Step 7 — Recognizer demo (optional)

If the user wants to see the active-learning loop too:

```bash
python tracking_eval/run_recognizer.py \
    --video <user's clip> --weights <user's .pt>
```

This runs two passes:
1. Empty bank → all events route UNKNOWN, persisted to
   `runs/recognizer_demo/<basename>_recognizer.db`.
2. Auto-label clusters → second pass shows KNOWN / AMBIGUOUS / UNKNOWN
   distribution as the bank fills.

For a real triage session use:

```bash
python -m combat_tracker_recognizer review <session_id>
```

-----

## What you should **not** do

- Do **not** replace the bundled `tracking/` modules with custom
  re-implementations. They are unit-tested; rewriting from scratch
  loses the test coverage.
- Do **not** silently change function signatures. If you must extend
  one (e.g. add a kwarg), keep existing keyword args back-compatible.
- Do **not** delete the `combat_tracker_recognizer` package even if
  the user only asked about tracking — it's wired into Phase 2/3 of
  the build plan and may be used later.
- Do **not** edit `combat_tracker_recognizer/DEVIATIONS.md` —
  it's the historical record of decisions; new deviations get
  appended.
- Do **not** push changes to the user's branch without an explicit
  request.
- Do **not** commit large binary files (model weights, eval videos)
  to git. Use `.gitignore`.

-----

## Common breakage and what to do

| symptom | likely cause | fix |
|---|---|---|
| `ImportError: tracking` | bundle dropped in wrong place | move `tracking/` to project root |
| `KeyError: 6` in `_split_detections` | class id mismatch | update `_PERSON_CLASS_PVP` etc. (Step 3) |
| `frames_with_masks: 0` despite `--mask-mode yolo_seg` | seg model returns shape `extract_yolo_seg_masks` doesn't recognise | update the shape parsing in `tracking/masks.py:extract_yolo_seg_masks` |
| `throws_total: 0` everywhere | class names not in the strike set | update `_PVP_CLASS_NAMES` so action classes use the recognised strings |
| `hit_rate: 1.0` everywhere | `target_id` resolved to closest non-owner without geometric overlap | (already fixed in current bundle — `attribution.landed` is checked) |
| pytest fails on `test_*_mask.py` | `lz4` / `mediapipe` not installed | `pip install -r requirements.txt` |

-----

## Reference: critical files in the bundle

When the user says "the tracker", they mean these:

- `tracking/pvp.py:PvPTracker.update` — main per-frame entry point.
  Optional `masks_per_person` arg threads instance masks through to
  the bank update (cleaner ReID) and the clinch detector (mask-IoU).
- `tracking/ownership.py:ActionOwnership.assign` — three-tier
  precedence: mask_iou → kinematic-chain → centroid. Returns an
  `AttributedAction` with a `landed` boolean (requires real geometric
  contact, not just proximity).
- `tracking/analytics.py:FighterAnalytics` — per-fighter accumulators.
  `record_action_thrown` / `record_action_landed` / `observe_frame`.
  Single-fighter scenes work too (slot 1 only).

When the user says "the recognizer", they mean
`combat_tracker_recognizer/recognizer.py:SubclassActionRecognizer`.

When the user asks about the build plan, the per-phase summaries are
in `combat_tracker_recognizer/PHASE_*_SUMMARY.md`.
