"""End-to-end recognizer demo on the sample PvP clip.

Runs the parent tracker (tracking/) over the video. For every YOLO action
detection (cross / hook / kick / high-guard / low-guard) it builds an
AttributedAction via tracking.ownership.ActionOwnership, then pipes that
event into combat_tracker_recognizer.CombatTrackerEventConsumer.

Two passes:

  Pass 1 — empty bank → all attributable actions persist as UNKNOWN.
  Pass 2 — open a ReviewSession, label every cluster with a placeholder
           subclass derived from its parent class (e.g. punch → punch_a),
           commit, then re-run and show how many events now route KNOWN.

Outputs:

  runs/recognizer_demo/<basename>_pass1_results.csv
  runs/recognizer_demo/<basename>_clusters.txt
  runs/recognizer_demo/<basename>_pass2_results.csv
  runs/recognizer_demo/<basename>_summary.json

Usage:
  python tracking_eval/run_recognizer.py \
      --video "Combat Sports Automation PvP/data/12.mp4" \
      --weights "Combat Sports Automation PvP/models/best.pt"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_THESIS_ROOT = _HERE.parent
if str(_THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_THESIS_ROOT))

from combat_tracker_recognizer.config import GateConfig, RecognizerConfig  # noqa: E402
from combat_tracker_recognizer.integration import CombatTrackerEventConsumer  # noqa: E402
from combat_tracker_recognizer.recognizer import SubclassActionRecognizer  # noqa: E402
from combat_tracker_recognizer.review.session import ReviewSession  # noqa: E402
from tracking.occlusion import ClinchDetector  # noqa: E402
from tracking.ownership import ActionOwnership  # noqa: E402
from tracking.person_filter import PersonFilter  # noqa: E402
from tracking.pvp import PvPTracker  # noqa: E402

# Reuse the loader from the qualitative runner.
from tracking_eval.run_qualitative import _load_yolo, _try_load_pose, _yolo_detections  # noqa: E402


_PERSON_CLASS_PVP = 6
_PVP_ACTION_NAMES = {1: "cross", 2: "high-guard", 3: "hook", 4: "kick", 5: "low-guard"}
_PARENT_CLASS_OF = {
    "cross": "punch", "hook": "punch",
    "kick": "kick",
    "high-guard": "guard", "low-guard": "guard",
}

_LANDMARK_NAMES = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
_MP_INDEX = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _split_dets(dets: np.ndarray):
    persons, actions = [], []
    for d in dets:
        x1, y1, x2, y2, conf, cls = d
        if conf < 0.3:
            continue
        box = (int(x1), int(y1), int(x2), int(y2))
        cid = int(cls)
        if cid == _PERSON_CLASS_PVP:
            persons.append(box)
        elif cid in _PVP_ACTION_NAMES:
            actions.append((cid, box, float(conf)))
    return persons, actions


def _whole_frame_landmarks(pose, frame_bgr):
    """Run MediaPipe on the whole frame and return one landmark dict (or None).

    The legacy ``mediapipe.solutions.pose`` only ever surfaces a single person.
    We re-broadcast that single skeleton to whichever person bbox it sits
    inside via ``_assign_landmarks_to_persons``.
    """
    if pose is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    out = {}
    for name, idx in _MP_INDEX.items():
        lm = res.pose_landmarks.landmark[idx]
        out[name] = (float(lm.x), float(lm.y), float(lm.visibility))
    return out


def _assign_landmarks_to_persons(landmarks, person_dets, frame_w, frame_h):
    """Assign whole-frame landmarks to whichever person bbox contains the
    nose+hip centroid. Other bboxes get None."""
    out = [None] * len(person_dets)
    if landmarks is None:
        return out
    # Build a centroid in pixel coords from a few stable landmarks.
    pts = []
    for name in ("nose", "left_hip", "right_hip"):
        lm = landmarks.get(name)
        if lm is None or lm[2] < 0.3:
            continue
        pts.append((lm[0] * frame_w, lm[1] * frame_h))
    if not pts:
        return out
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    for i, (x1, y1, x2, y2) in enumerate(person_dets):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out[i] = landmarks
            return out
    return out


def _track_landmarks_for_slot(slot_bbox, person_dets, landmarks_per_person):
    """Greedy match: pick the input detection with highest IoU against the
    tracker's slot bbox; return its landmarks."""
    if slot_bbox is None or not person_dets:
        return None
    from tracking.kalman import iou
    best_iou, best_idx = -1.0, -1
    for i, b in enumerate(person_dets):
        s = iou(slot_bbox, b)
        if s > best_iou:
            best_iou, best_idx = s, i
    if best_iou < 0.05 or best_idx < 0:
        return None
    return landmarks_per_person[best_idx]


def run_one_pass(
    video_path: str,
    weights: str,
    recognizer: SubclassActionRecognizer,
    session_id: str,
    log_path: str,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = _load_yolo(weights)
    pose = _try_load_pose()
    tracker = PvPTracker()
    pfilter = PersonFilter()
    clinch = ClinchDetector()
    ownership = ActionOwnership(clinch_detector=clinch)
    consumer = CombatTrackerEventConsumer(
        recognizer=recognizer, window_before=8, window_after=4,
        session_id=session_id, fps=fps,
    )

    rows = []
    fi = 0
    decision_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = _yolo_detections(model, frame)
        person_raw, actions = _split_dets(dets)
        person, _, _ = pfilter.filter(person_raw)
        whole_frame_lm = _whole_frame_landmarks(pose, frame)
        landmarks_per_person = _assign_landmarks_to_persons(
            whole_frame_lm, person, w, h,
        )

        tracker.update(frame, person, landmarks_per_person)
        slot_boxes = {tid: tracker.slots[tid].bbox for tid in ("1", "2")}
        clinch.observe(fi, slot_boxes,
                       num_person_detections=len(person),
                       person_dets=person)

        # Push landmarks per tracked slot.
        for tid in ("1", "2"):
            sb = tracker.slots[tid].bbox
            lm = _track_landmarks_for_slot(sb, person, landmarks_per_person)
            consumer.push_track_keypoints(fi, int(tid), lm)

        # Attribute each action and feed the consumer.
        tracks_for_ownership = {
            tid: {"box": tracker.slots[tid].bbox} for tid in ("1", "2")
        }
        landmarks_for_ownership = {
            tid: _track_landmarks_for_slot(tracker.slots[tid].bbox, person,
                                           landmarks_per_person)
            for tid in ("1", "2")
        }
        for cid, action_box, conf in actions:
            action_name = _PVP_ACTION_NAMES[cid]
            attribution = ownership.assign(
                action_id=fi * 10 + cid,
                action_class=action_name,
                action_box=action_box,
                frame_idx=fi,
                frame_size=(w, h),
                tracks=tracks_for_ownership,
                landmarks_per_track=landmarks_for_ownership,
            )
            if attribution.owner_id is None:
                continue
            # Map action_class to parent_class for the recognizer.
            class _RemappedAttribution:
                def __init__(self, orig, parent):
                    self.action_class = parent
                    self.owner_id = orig.owner_id
                    self.target_id = orig.target_id

            parent = _PARENT_CLASS_OF.get(action_name, action_name)
            wrapped = _RemappedAttribution(attribution, parent)
            result = consumer.observe_action(wrapped, frame_idx=fi,
                                              video_ref=video_path)
            if result is not None:
                decision_counts[result.decision.value] += 1
                parent_counts[parent] += 1
                rows.append({
                    "frame": fi,
                    "yolo_action": action_name,
                    "parent_class": parent,
                    "owner": attribution.owner_id,
                    "decision": result.decision.value,
                    "subclass": result.subclass or "",
                    "confidence": f"{result.confidence:.3f}",
                    "clip_id": result.clip_id if result.clip_id is not None else "",
                    "top_match": (result.top_matches[0][0] + ":"
                                  + f"{result.top_matches[0][1]:.3f}") if result.top_matches else "",
                })
        # Drain queued actions whose window_after deadline has arrived.
        for r in consumer.tick(fi):
            decision_counts[r.decision.value] += 1
            rows.append({
                "frame": fi,
                "yolo_action": "(drained)",
                "parent_class": "",
                "owner": "",
                "decision": r.decision.value,
                "subclass": r.subclass or "",
                "confidence": f"{r.confidence:.3f}",
                "clip_id": r.clip_id if r.clip_id is not None else "",
                "top_match": "",
            })
        fi += 1
    cap.release()
    for r in consumer.flush():
        decision_counts[r.decision.value] += 1
        rows.append({
            "frame": fi,
            "yolo_action": "(flush)",
            "parent_class": "",
            "owner": "",
            "decision": r.decision.value,
            "subclass": r.subclass or "",
            "confidence": f"{r.confidence:.3f}",
            "clip_id": r.clip_id if r.clip_id is not None else "",
            "top_match": "",
        })

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "frame", "yolo_action", "parent_class", "owner", "decision",
            "subclass", "confidence", "clip_id", "top_match",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return {
        "frames_processed": fi,
        "events_recorded": len(rows),
        "decision_counts": dict(decision_counts),
        "events_by_parent": dict(parent_counts),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", default="runs/recognizer_demo")
    args = p.parse_args()

    base = Path(args.video).stem
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = RecognizerConfig()
    cfg.store.db_path = os.path.join(args.out_dir, f"{base}_recognizer.db")
    # Tighten gate for the random-init encoder (per DEVIATIONS.md D3).
    cfg.gate = GateConfig(
        known_distance_threshold=1e-4,
        ambiguous_distance_threshold=5e-4,
        min_margin_ratio=2.0,
    )
    # Lower the cluster size so a small demo can produce real clusters.
    cfg.review.min_cluster_size = 2

    # Fresh DB every run.
    if os.path.exists(cfg.store.db_path):
        os.remove(cfg.store.db_path)

    recognizer = SubclassActionRecognizer(cfg)

    print("=== PASS 1: empty bank ===")
    pass1 = run_one_pass(
        args.video, args.weights, recognizer,
        session_id="pass1",
        log_path=os.path.join(args.out_dir, f"{base}_pass1_results.csv"),
    )
    print(json.dumps(pass1, indent=2))

    # Cluster the unlabeled clips and label every cluster with a placeholder
    # subclass per parent. Real review work would inspect with `play` and
    # apply meaningful names.
    print("\n=== Auto-labelling captured clusters ===")
    sess = ReviewSession(
        session_id="auto", clipstore=recognizer.clipstore,
        bank=recognizer.bank, config=cfg.review,
        encoder_version=recognizer.encoder.version,
    )
    cluster_lines = []
    parent_label_counter: Counter[str] = Counter()
    clusters = sess.list_clusters()
    # Iterate over a stable snapshot. Each label_cluster invalidates the
    # cluster cache; if we relied on the session's get_cluster between
    # iterations the cluster ids would shift mid-loop. Call the bank +
    # clipstore directly using the snapshot's member ids.
    for c in clusters:
        parent_label_counter[c.parent_class] += 1
        idx = parent_label_counter[c.parent_class]
        # Zero-padded two-digit suffix avoids unicode overflow when there
        # are more than 26 clusters per parent (the random-init encoder
        # produces lots of singletons on small samples — see DEVIATIONS.md
        # D3).
        subclass = f"{c.parent_class}_{idx:02d}"
        for clip_id in c.member_clip_ids:
            clip = recognizer.clipstore.get_clip(clip_id)
            recognizer.clipstore.label_clip(
                clip_id, subclass, c.parent_class,
                labeled_by="auto", session_id="auto",
            )
            recognizer.bank.add(c.parent_class, subclass, clip.embedding.vector,
                                encoder_version=recognizer.encoder.version)
        cluster_lines.append(
            f"cluster {c.id:3d}  parent={c.parent_class:6s}  size={c.size:3d}  "
            f"intra={c.intra_distance_mean:.4f}  -> {subclass}"
        )
    sess.commit(note="auto-labelled in run_recognizer demo")

    cluster_path = os.path.join(args.out_dir, f"{base}_clusters.txt")
    with open(cluster_path, "w") as f:
        f.write("\n".join(cluster_lines) + "\n")
    print(f"labelled {len(clusters)} cluster(s); see {cluster_path}")

    print("\n=== PASS 2: bank seeded with auto-labels ===")
    # Wipe the clip side of the DB so pass-2 only persists NEW unknowns —
    # labels and bank survive in-memory because we never closed `recognizer`.
    pass2 = run_one_pass(
        args.video, args.weights, recognizer,
        session_id="pass2",
        log_path=os.path.join(args.out_dir, f"{base}_pass2_results.csv"),
    )
    print(json.dumps(pass2, indent=2))

    summary = {
        "video": args.video,
        "pass1": pass1,
        "clusters_labelled": len(clusters),
        "pass2": pass2,
        "bank_status": recognizer.bank_status(),
    }
    with open(os.path.join(args.out_dir, f"{base}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
