"""Standalone qualitative eval for the new tracker.

No Streamlit, no thesis-app coupling. Loads YOLO via torch.hub the same way
the apps do, runs the new tracker on a video, and writes:
  - <basename>_v2_overlay.mp4 with id-coloured boxes + clinch banner
  - <basename>_v2_log.csv  with per-frame track + clinch state
  - <basename>_v2_summary.json with totals (frames, ids assigned, clinch events)

Usage:
  python tracking_eval/run_qualitative.py \
      --video "Combat Sports Automation PvP/data/12.mp4" \
      --weights "Combat Sports Automation PvP/models/best.pt" \
      --mode pvp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the package importable when run from the repo root.
_HERE = Path(__file__).resolve().parent
_THESIS_ROOT = _HERE.parent
if str(_THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_THESIS_ROOT))

from tracking.analytics import FighterAnalytics  # noqa: E402
from tracking.masks import extract_yolo_seg_masks, grabcut_mask, mask_overlap_fraction  # noqa: E402
from tracking.occlusion import ClinchDetector  # noqa: E402
from tracking.ownership import ActionOwnership  # noqa: E402
from tracking.person_filter import PersonFilter  # noqa: E402
from tracking.pve import PvETracker  # noqa: E402
from tracking.pvp import PvPTracker  # noqa: E402


_PERSON_CLASS_PVE = 4
_BAG_CLASS_PVE = 0
_PERSON_CLASS_PVP = 6
_BAG_CLASS_PVP = 0  # boxing-bag in both class lists

_ID_COLOURS = {"1": (0, 255, 255), "2": (255, 0, 255), "bag": (0, 200, 255)}


def _load_yolo(weights: str):
    """Load a YOLOv5 .pt checkpoint.

    Prefers torch.hub (matches the Streamlit apps). Falls back to the
    ``yolov5`` PyPI package if hub is unreachable. PyTorch 2.6+ defaults
    ``weights_only=True`` which rejects legacy v5 pickles, so we patch
    torch.load for the duration of the load call only.
    """
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


def _try_load_pose():
    """Optional MediaPipe pose. Returns None if unavailable."""
    try:
        import mediapipe as mp
        return mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.4, min_tracking_confidence=0.4,
        )
    except Exception:
        return None


_LANDMARK_INDEX = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _extract_landmarks(pose, frame_bgr) -> dict | None:
    if pose is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return None
    out = {}
    for name, idx in _LANDMARK_INDEX.items():
        lm = results.pose_landmarks.landmark[idx]
        out[name] = (float(lm.x), float(lm.y), float(lm.visibility))
    return out


def _yolo_detections(model, frame_bgr) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    model.iou = 0.2
    model.conf = 0.2
    res = model(rgb)
    return res.xyxy[0].cpu().numpy()


def _split_detections(dets: np.ndarray, person_cls: int, bag_cls: int):
    person, bag, actions = [], [], []
    for d in dets:
        x1, y1, x2, y2, conf, cls_id = d
        if conf < 0.3:
            continue
        box = (int(x1), int(y1), int(x2), int(y2))
        cid = int(cls_id)
        if cid == person_cls:
            person.append(box)
        elif cid == bag_cls:
            bag.append(box)
        else:
            actions.append((cid, box, float(conf)))
    return person, bag, actions


# Class-name maps lifted from the Streamlit apps.
_PVE_CLASS_NAMES = {
    0: "boxing-bag", 1: "high-guard", 2: "kick-knee", 3: "low-guard",
    4: "person", 5: "punch",
}
_PVP_CLASS_NAMES = {
    0: "boxing-bag", 1: "cross", 2: "high-guard", 3: "hook",
    4: "kick", 5: "low-guard", 6: "person",
}
_ACTION_COLOURS = {
    "cross": (255, 0, 255), "hook": (255, 165, 0), "kick": (0, 255, 255),
    "punch": (255, 0, 255), "kick-knee": (0, 255, 255),
    "high-guard": (127, 255, 127), "low-guard": (255, 255, 0),
}


def _draw_actions(frame, actions: list, names_map: dict) -> None:
    for cid, box, conf in actions:
        name = names_map.get(cid, str(cid))
        color = _ACTION_COLOURS.get(name, (200, 200, 200))
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (box[0], min(frame.shape[0] - 5, box[3] + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _draw_track(frame, tid, box, conf=None, clinch=False):
    if box is None:
        return
    color = _ID_COLOURS.get(tid, (255, 255, 255))
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    label = f"ID {tid}"
    if conf is not None:
        label += f" {conf:.2f}"
    if clinch:
        label += " [CLINCH]"
    cv2.putText(frame, label, (box[0], max(14, box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if clinch else color, 2)


def _draw_mask_outline(frame, mask, color):
    """Draw the mask edge so the user can see the segmentation."""
    if mask is None:
        return
    contours, _ = cv2.findContours((mask.astype(np.uint8) > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    cv2.drawContours(frame, contours, -1, color, 2)


def _draw_mask_translucent(frame, mask, color, alpha=0.25):
    """Tint the mask region for a visible mask vs bbox difference."""
    if mask is None:
        return
    overlay = frame.copy()
    overlay[mask.astype(bool)] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def _draw_metrics_panel(frame, summary):
    """Bottom-left HUD: throws, hit rate, clinch time per fighter."""
    h, w = frame.shape[:2]
    pad = 8
    line_h = 18
    lines = []
    for tid in ("1", "2"):
        f = summary["fighters"][tid]
        lines.append(
            f"ID {tid}  thr {f['throws_total']:>3d}  land {f['landed_total']:>2d} "
            f"hit% {f['hit_rate']*100:>4.0f}  clinch {f['time_in_clinch_seconds']:>4.1f}s "
            f"trav {f['travel_distance_px']:>5.0f}px"
        )
    lines.append(
        f"engage avg {summary['engagement']['mean_distance_between_fighters_px']:>4.0f}px  "
        f"in-range {summary['engagement']['frames_within_strike_range']:>3d}f"
    )
    box_h = line_h * len(lines) + pad * 2
    box_w = 480
    x0, y0 = pad, h - box_h - pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + pad, y0 + pad + line_h * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def _per_person_masks(model_results, frame, person_dets, frame_shape,
                      mask_mode: str) -> Optional[list]:
    """Build per-detection masks. Returns None when ``mask_mode='none'`` or
    no mask source is available. Order matches ``person_dets``.

    ``mask_mode`` values:
        - "none": no masks (back-compat).
        - "yolo_seg": pull from a YOLOv26-seg results object. Returns None
          if the model is detection-only.
        - "grabcut": synthesise per-bbox masks via OpenCV GrabCut. Slow
          (~50-200 ms/frame total) but useful as a stand-in until a
          segmentation model is trained.
    """
    if mask_mode == "none":
        return None
    if mask_mode == "yolo_seg":
        return extract_yolo_seg_masks(model_results, frame_shape)
    if mask_mode == "grabcut":
        out = []
        for box in person_dets:
            m = grabcut_mask(frame, box, num_iters=2)
            out.append(m)
        return out
    raise ValueError(f"unknown mask_mode: {mask_mode}")


def run_pvp(video_path: str, weights: str, out_dir: str,
            mask_mode: str = "none",
            max_frames: Optional[int] = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    base = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = _load_yolo(weights)
    pose = _try_load_pose()

    tracker = PvPTracker()
    clinch = ClinchDetector()
    person_filter = PersonFilter()

    overlay_path = os.path.join(out_dir, f"{base}_v2_overlay.mp4")
    csv_path = os.path.join(out_dir, f"{base}_v2_log.csv")
    summary_path = os.path.join(out_dir, f"{base}_v2_summary.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    log_rows: list[dict] = []
    clinch_events = 0
    in_clinch = False
    fi = 0
    mask_frames = 0  # number of frames where any per-person mask was used
    analytics = FighterAnalytics(fps=fps, frame_size=(w, h))
    ownership = ActionOwnership(clinch_detector=clinch)
    next_action_id = 0

    while True:
        if max_frames is not None and fi >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        dets = _yolo_detections(model, frame)
        person_raw, _bag, actions = _split_detections(dets, _PERSON_CLASS_PVP, _BAG_CLASS_PVP)

        # Pre-filter referees / background. landmarks computed AFTER filtering
        # to avoid wasting MediaPipe inference on dropped detections.
        person, _, filter_decisions = person_filter.filter(person_raw)

        # Per-person masks (optional).
        masks_per_person = _per_person_masks(
            None, frame, person, frame.shape[:2], mask_mode,
        )
        if masks_per_person and any(m is not None for m in masks_per_person):
            mask_frames += 1

        landmarks_per_person: list = [None] * len(person)
        if pose is not None and person:
            # Run pose on largest person bbox crop. Cheap heuristic; good enough for eval.
            for i, box in enumerate(person):
                x1, y1, x2, y2 = box
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                lm = _extract_landmarks(pose, crop)
                if lm is None:
                    continue
                # Re-normalise crop landmarks to full-frame coords.
                cw, ch = (x2 - x1), (y2 - y1)
                landmarks_per_person[i] = {
                    name: ((x1 + lm[name][0] * cw) / w,
                           (y1 + lm[name][1] * ch) / h,
                           lm[name][2]) for name in lm
                }

        tracker.update(frame, person, landmarks_per_person,
                       masks_per_person=masks_per_person)
        slot_boxes = {tid: tracker.slots[tid].bbox for tid in ("1", "2")}
        slot_masks = {tid: tracker.slots[tid].last_mask for tid in ("1", "2")}
        state = clinch.observe(fi, slot_boxes, num_person_detections=len(person))
        if state.active and not in_clinch:
            clinch_events += 1
            in_clinch = True
        elif not state.active:
            in_clinch = False

        # Map detection-index landmarks to slot ids (which detection went to which slot).
        slot_landmarks: dict[str, Optional[dict]] = {"1": None, "2": None}
        for tid in ("1", "2"):
            sb = tracker.slots[tid].bbox
            if sb is None or not person:
                continue
            best_iou, best_idx = -1.0, -1
            for i, pb in enumerate(person):
                from tracking.kalman import iou as _iou
                v = _iou(sb, pb)
                if v > best_iou:
                    best_iou, best_idx = v, i
            if best_iou >= 0.3 and 0 <= best_idx < len(landmarks_per_person):
                slot_landmarks[tid] = landmarks_per_person[best_idx]

        # Per-action ownership + analytics.
        action_classes_present = []
        for cid, abox, _conf in actions:
            cname = _PVP_CLASS_NAMES.get(cid, str(cid))
            action_classes_present.append(cname)
            if cname in ("high-guard", "low-guard"):
                continue  # guards are stance, not strikes
            attribution = ownership.assign(
                action_id=next_action_id, action_class=cname, action_box=abox,
                frame_idx=fi, frame_size=(w, h),
                tracks={tid: {"box": tracker.slots[tid].bbox} for tid in ("1", "2")},
                landmarks_per_track=slot_landmarks,
            )
            next_action_id += 1
            if attribution.owner_id is not None:
                analytics.record_action_thrown(attribution.owner_id, cname)
                if attribution.landed and attribution.target_id \
                        and attribution.target_id != attribution.owner_id:
                    analytics.record_action_landed(
                        attribution.owner_id, attribution.target_id, cname,
                    )

        analytics.observe_frame(
            slot_bboxes=slot_boxes,
            slot_landmarks=slot_landmarks,
            clinch_active=state.active,
            action_classes_present=action_classes_present,
        )

        # ---- Draw layer order: masks first (under bboxes), then bboxes/HUD.
        if masks_per_person:
            for box, m in zip(person, masks_per_person):
                if m is None:
                    continue
                _draw_mask_translucent(frame, m, (60, 200, 255), alpha=0.20)
                _draw_mask_outline(frame, m, (0, 220, 255))

        _draw_actions(frame, actions, _PVP_CLASS_NAMES)
        for d in filter_decisions:
            if d.reason == "kept":
                continue
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
            cv2.putText(frame, f"drop: {d.reason}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        for tid in ("1", "2"):
            _draw_track(frame, tid, tracker.slots[tid].bbox, clinch=state.active)
        _draw_metrics_panel(frame, analytics.summary())
        writer.write(frame)

        log_rows.append({
            "frame": fi,
            "num_person_dets": len(person),
            "id1_box": tracker.slots["1"].bbox,
            "id2_box": tracker.slots["2"].bbox,
            "clinch": int(state.active),
        })
        fi += 1

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "num_person_dets", "id1_box", "id2_box", "clinch"])
        for row in log_rows:
            writer_csv.writerow([row["frame"], row["num_person_dets"],
                                 row["id1_box"], row["id2_box"], row["clinch"]])

    analytics_summary = analytics.summary()
    summary = {
        "video": video_path,
        "frames": fi,
        "id1_visible_frames": sum(1 for r in log_rows if r["id1_box"] is not None),
        "id2_visible_frames": sum(1 for r in log_rows if r["id2_box"] is not None),
        "clinch_events": clinch_events,
        "anchored": tracker.anchored,
        "mask_mode": mask_mode,
        "frames_with_masks": mask_frames,
        "analytics": analytics_summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_pve(video_path: str, weights: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    base = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = _load_yolo(weights)
    tracker = PvETracker()
    person_filter = PersonFilter()

    overlay_path = os.path.join(out_dir, f"{base}_v2_overlay.mp4")
    csv_path = os.path.join(out_dir, f"{base}_v2_log.csv")
    summary_path = os.path.join(out_dir, f"{base}_v2_summary.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    log_rows = []
    fi = 0
    bag_state_counts = {"resting": 0, "swinging": 0, "struck": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = _yolo_detections(model, frame)
        person_raw, bag, actions = _split_detections(dets, _PERSON_CLASS_PVE, _BAG_CLASS_PVE)
        person, _, _ = person_filter.filter(person_raw)
        state = tracker.update(fi, person, bag)
        bag_state_counts[state["bag_state"]] = bag_state_counts.get(state["bag_state"], 0) + 1

        _draw_actions(frame, actions, _PVE_CLASS_NAMES)
        if state["person_track"] is not None:
            _draw_track(frame, "1", state["person_track"].bbox)
        if state["bag_track"] is not None:
            _draw_track(frame, "bag", state["bag_track"].bbox)
        cv2.putText(frame, f"bag: {state['bag_state']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        writer.write(frame)

        log_rows.append({
            "frame": fi,
            "person_box": state["person_track"].bbox if state["person_track"] else None,
            "bag_box": state["bag_track"].bbox if state["bag_track"] else None,
            "bag_state": state["bag_state"],
        })
        fi += 1

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "person_box", "bag_box", "bag_state"])
        for r in log_rows:
            wcsv.writerow([r["frame"], r["person_box"], r["bag_box"], r["bag_state"]])

    summary = {
        "video": video_path,
        "frames": fi,
        "person_visible_frames": sum(1 for r in log_rows if r["person_box"] is not None),
        "bag_visible_frames": sum(1 for r in log_rows if r["bag_box"] is not None),
        "bag_state_distribution": bag_state_counts,
        "warnings": tracker.warnings,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--mode", choices=("pve", "pvp"), required=True)
    p.add_argument("--out-dir", default="runs/tracking_v2")
    p.add_argument(
        "--mask-mode", choices=("none", "yolo_seg", "grabcut"), default="none",
        help=("Mask source. 'none' = bbox-only (legacy). 'yolo_seg' = pull "
              "masks from a YOLO segmentation results object. 'grabcut' = "
              "synthesise per-bbox masks via GrabCut (slow, no training "
              "required, useful for A/B comparison)."),
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap the number of frames processed; useful for slow modes like grabcut.",
    )
    args = p.parse_args()
    if args.mode == "pvp":
        summary = run_pvp(args.video, args.weights, args.out_dir,
                          mask_mode=args.mask_mode,
                          max_frames=args.max_frames)
    else:
        summary = run_pve(args.video, args.weights, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
