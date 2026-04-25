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

from tracking.occlusion import ClinchDetector  # noqa: E402
from tracking.ownership import ActionOwnership  # noqa: E402
from tracking.pve import PvETracker  # noqa: E402
from tracking.pvp import PvPTracker  # noqa: E402


_PERSON_CLASS_PVE = 4
_BAG_CLASS_PVE = 0
_PERSON_CLASS_PVP = 6
_BAG_CLASS_PVP = 0  # boxing-bag in both class lists

_ID_COLOURS = {"1": (0, 255, 255), "2": (255, 0, 255), "bag": (0, 200, 255)}


def _load_yolo(weights: str):
    import torch
    return torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=False)


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
    person, bag = [], []
    for d in dets:
        x1, y1, x2, y2, conf, cls_id = d
        if conf < 0.3:
            continue
        box = (int(x1), int(y1), int(x2), int(y2))
        if int(cls_id) == person_cls:
            person.append(box)
        elif int(cls_id) == bag_cls:
            bag.append(box)
    return person, bag


def _draw_track(frame, tid, box, conf=None):
    if box is None:
        return
    color = _ID_COLOURS.get(tid, (255, 255, 255))
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    label = f"ID {tid}"
    if conf is not None:
        label += f" {conf:.2f}"
    cv2.putText(frame, label, (box[0], max(0, box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_pvp(video_path: str, weights: str, out_dir: str) -> dict:
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

    overlay_path = os.path.join(out_dir, f"{base}_v2_overlay.mp4")
    csv_path = os.path.join(out_dir, f"{base}_v2_log.csv")
    summary_path = os.path.join(out_dir, f"{base}_v2_summary.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    log_rows: list[dict] = []
    clinch_events = 0
    in_clinch = False
    fi = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = _yolo_detections(model, frame)
        person, _bag = _split_detections(dets, _PERSON_CLASS_PVP, _BAG_CLASS_PVP)

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

        tracker.update(frame, person, landmarks_per_person)
        slot_boxes = {tid: tracker.slots[tid].bbox for tid in ("1", "2")}
        state = clinch.observe(fi, slot_boxes, num_person_detections=len(person))
        if state.active and not in_clinch:
            clinch_events += 1
            in_clinch = True
        elif not state.active:
            in_clinch = False

        for tid in ("1", "2"):
            _draw_track(frame, tid, tracker.slots[tid].bbox)
        if state.active:
            cv2.putText(frame, "CLINCH (suppress ownership)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
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

    summary = {
        "video": video_path,
        "frames": fi,
        "id1_visible_frames": sum(1 for r in log_rows if r["id1_box"] is not None),
        "id2_visible_frames": sum(1 for r in log_rows if r["id2_box"] is not None),
        "clinch_events": clinch_events,
        "anchored": tracker.anchored,
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
        person, bag = _split_detections(dets, _PERSON_CLASS_PVE, _BAG_CLASS_PVE)
        state = tracker.update(fi, person, bag)
        bag_state_counts[state["bag_state"]] = bag_state_counts.get(state["bag_state"], 0) + 1

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
    args = p.parse_args()
    if args.mode == "pvp":
        summary = run_pvp(args.video, args.weights, args.out_dir)
    else:
        summary = run_pve(args.video, args.weights, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
