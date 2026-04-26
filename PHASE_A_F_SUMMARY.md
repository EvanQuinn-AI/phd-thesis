# Phase A–F summary: mask-aware tracker upgrade

Implements the "Future Architecture: YOLOv26 Instance Segmentation × Pose
× Bbox" section of the plan. Each phase is independently testable and
each phase is gated by `mask=None` defaults so the bbox-only path
remains the back-compat backstop.

## Phase A — mask helpers (`tracking/masks.py`)

- `mask_iou(m_a, m_b)` — instance mask IoU.
- `mask_overlap_fraction(m_a, m_b)` — share of `m_a` pixels also in
  `m_b`. Used by the action-ownership rule (action mask sits mostly
  inside the puncher's mask).
- `bbox_to_mask(bbox, frame_shape)` — degenerate rectangle mask.
- `grabcut_mask(frame, bbox, num_iters)` — OpenCV-based foreground
  mask. ~535 ms/bbox in this sandbox; used for the demo without a
  trained seg model.
- `extract_yolo_seg_masks(results, frame_shape)` — adapter that pulls
  per-detection masks out of an Ultralytics seg-model results object;
  returns `None` for legacy detection-only results.

## Phase B — mask-aware `PartExtractor`

`PartExtractor.extract(frame, bbox, landmarks, mask=None)`. When `mask`
is provided, every region's bounding sub-rectangle is intersected with
the mask before the HSV histogram is computed, removing background and
other-fighter pixels. Test
`test_features_mask.py::test_mask_strips_background_from_torso_histogram`
asserts the masked histogram of a colour-straddling bbox matches a
clean reference better than the unmasked one.

## Phase C — mask-IoU `ClinchDetector`

`ClinchDetector.observe` accepts optional `slot_masks` (predicted slot
masks) and `person_masks` (raw detection masks). When present, mask-IoU
replaces bbox-IoU on both the entry trigger and the detection-driven
exit trigger. Bbox path stays the default. Tests:

- `test_bbox_overlap_does_not_trigger_when_masks_separate` — two bboxes
  fully overlap, masks disjoint, no clinch.
- `test_mask_overlap_does_trigger_clinch` — masks share ~half their
  pixels, clinch fires.
- `test_mask_path_falls_back_to_bbox_when_masks_missing`.
- `test_mask_det_separation_drives_exit`.

## Phase D — `PvPTracker` end-to-end

`PvPTracker.update(frame, person_dets, landmarks_per_person,
masks_per_person=None)`. `PvPSlot` gains `last_mask`. Masks are threaded
to `PartExtractor` for bank updates and to `ClinchDetector` for
predicted-slot mask-IoU. The legacy 8-test PvP suite still passes with
`masks_per_person=None`; new tests in `test_pvp_mask.py`:

- `test_pvp_tracker_runs_with_masks` — full anchor-and-track run with
  masks supplied.
- `test_pvp_tracker_mask_argument_is_optional` — back-compat.
- `test_mask_path_keeps_clean_feature_bank` — bbox-overlapping fighters
  with mask-separated banks: a fresh red sample matches slot 1 (red
  bank) better than slot 2 (blue bank).

## Phase E — three-tier `ActionOwnership`

`ActionOwnership.assign(..., action_mask=None, masks_per_track=None,
bag_mask=None)`. Precedence:

1. **mask_iou** — owner = `arg max_k overlap_fraction(action_mask, mask_k)`.
2. **kinematic** — terminal-keypoint distance to action centre.
3. **centroid_fallback** — bbox-contains-action-centroid.

Result's `method` field records which tier won. Target inference
similarly uses mask-overlap when masks are present; falls back to bbox
+ inverse-distance otherwise. Tests in `test_ownership_mask.py`:

- `test_mask_iou_owner_wins_when_bboxes_overlap` — straddle case,
  owner=puncher (more overlap), target=recipient (next).
- `test_mask_iou_takes_precedence_over_kinematic` — mask trumps pose.
- `test_mask_path_falls_back_to_kinematic_when_action_mask_missing`.
- `test_mask_target_resolution`.
- `test_mask_target_picks_bag_when_only_bag_mask_present`.

## Phase F — eval runner threads masks through

`tracking_eval/run_qualitative.py` gains `--mask-mode {none,
yolo_seg, grabcut}` and `--max-frames N`. The `_per_person_masks`
helper builds masks per detection from the chosen source and threads
them to `PvPTracker.update`.

## Results

`pytest tests/tracking/`: **73 passed**.
`pytest combat_tracker_recognizer/tests/`: **48 passed**.

Real-clip A/B at `tracking_eval/sample_outputs/mask_demo/`:

| metric | bbox-only | mask (grabcut) |
|---|---:|---:|
| frames processed | 100 | 100 |
| ID 1 / ID 2 visible | 71 / 71 | 71 / 71 |
| clinch events | 1 | 1 |
| frames with masks used | 0 | 100 |

Mask path runs end-to-end on real data with zero regression. Where the
mask path would *change* the routing — heavy bbox overlap with
separated bodies, action attribution where the centroid sits in both
fighters' bboxes — the unit tests exercise the win condition. A
trained YOLOv26-seg model would let the same harness produce a
full-clip A/B in seconds rather than minutes.

## Open follow-ups (from the plan)

1. Train (or fine-tune) a YOLOv26-seg model on the 6/7-class boxing
   dataset.
2. Re-run `tracking_eval/run_qualitative.py --mask-mode yolo_seg` on
   the full clip; produce ablation rows for chapter 6.
3. Section 5.5 Transformer-attribution head: extend the per-frame
   feature vector with mask area, mask-IoU per fighter, and
   mask-centroid velocity (cleaner than bbox-centroid).
