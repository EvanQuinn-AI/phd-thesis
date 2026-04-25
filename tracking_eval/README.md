# Qualitative tracking evaluation

Standalone runner for the new `tracking/` package. No Streamlit dependency,
no thesis-app coupling — just YOLO + the new tracker, against a video file.

## Run

```bash
# PvP (two fighters)
python tracking_eval/run_qualitative.py \
    --video "Combat Sports Automation PvP/data/12.mp4" \
    --weights "Combat Sports Automation PvP/models/best.pt" \
    --mode pvp \
    --out-dir runs/tracking_v2

# PvE (person + bag)
python tracking_eval/run_qualitative.py \
    --video <path/to/bag_clip.mp4> \
    --weights "Combat Sports Automation/models/best.pt" \
    --mode pve
```

## Outputs (under `--out-dir`)

- `<basename>_v2_overlay.mp4` — id-coloured boxes; PvP shows a "CLINCH"
  banner during clinch frames; PvE shows the rolling bag state
  (`resting | swinging | struck`).
- `<basename>_v2_log.csv` — per-frame track state.
- `<basename>_v2_summary.json` — totals: frames processed, frames each id
  was visible, clinch events (PvP), bag-state distribution (PvE).

## What to look for (no MOT ground truth available)

PvP:
- Both id colours stay attached to the same fighter across the whole clip.
  Spot-check by scrubbing the overlay video.
- Clinch banner appears during obvious grappling sequences.
- After the clinch resolves, the id colours match the same fighters they
  started on.

PvE:
- Person box stays on the trainee, bag box stays on the bag.
- `bag_state` flips to `struck` close to visible impacts.
- `summary.json#warnings` is empty (or contains the explicit "consider PvP
  mode" notice if a second person genuinely persists).

## Switching the Streamlit apps to v2

Both `Combat Sports Automation/` and `Combat Sports Automation PvP/`
gpu-version apps fall back to legacy behaviour by default. Opt in with:

```bash
USE_TRACKING_V2=1 streamlit run "Combat Sports Automation PvP/gpu-version/app.py"
```

The legacy `update_two_person_ids` / `find_action_owner` / `check_overlap`
paths remain in the source for the ablation comparison; the env var is the
only switch.
