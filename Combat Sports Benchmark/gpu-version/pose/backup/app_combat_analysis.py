# app_combat_analysis.py
# Combat Sports Analysis — YOLO Detection + Pose Biomechanics
#
# Architecture:
#   YOLO  → Object detection (bag, punch, kick, person, guard)
#         → Strike counting (overlap-based event detection)
#         → Benchmarked pipeline latencies (Thesis Table 4.2)
#
#   Pose  → Skeleton tracking (MediaPipe mp.solutions.pose)
#         → Joint velocities, accelerations, limb extensions
#         → Guard position monitoring
#         → Per-strike biomechanics correlated with YOLO detections

import os
import json
import logging
import warnings
import time
import pandas as pd
import numpy as np
import torch
import yaml
import streamlit as st
import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
from pose_analytics import PoseAnalytics

# ── Suppress warnings ──────────────────────────────────────────────────────
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
torch.classes.__path__ = []
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("torch").setLevel(logging.ERROR)
st.set_page_config(page_title="Combat Sports Analysis", layout="wide")


# =============================================================================
# BENCHMARKING (Thesis Table 4.2)
# =============================================================================

class PipelineBenchmark:
    """Collect timing measurements for each pipeline stage."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.stage_names = [
            'frame_extraction', 'yolo_inference', 'pose_inference',
            'hit_detection', 'overlay_rendering', 'total_pipeline'
        ]
        self.hardware_info = {}

    def record(self, stage: str, duration_ms: float):
        self.timings[stage].append(duration_ms)

    def capture_hardware_info(self):
        self.hardware_info = {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'pytorch_version': torch.__version__,
        }
        if torch.cuda.is_available():
            self.hardware_info['gpu_name'] = torch.cuda.get_device_name(0)
            self.hardware_info['cuda_version'] = torch.version.cuda
        return self.hardware_info

    def get_statistics(self):
        stats = {}
        for stage in self.stage_names:
            if self.timings[stage]:
                arr = np.array(self.timings[stage])
                stats[stage] = {
                    'mean': np.mean(arr), 'std': np.std(arr),
                    'median': np.median(arr),
                    'p95': np.percentile(arr, 95),
                    'p99': np.percentile(arr, 99),
                    'min': np.min(arr), 'max': np.max(arr),
                    'count': len(arr),
                }
            else:
                stats[stage] = None
        return stats

    def to_dataframe(self):
        stats = self.get_statistics()
        rows = []
        for stage in self.stage_names:
            if stats[stage]:
                s = stats[stage]
                name = stage.replace('_', ' ').title()
                if stage == 'yolo_inference':
                    name = 'YOLO Inference (640×640)'
                elif stage == 'pose_inference':
                    name = 'Pose Estimation'
                rows.append({
                    'Component': name,
                    'Mean_ms': round(s['mean'], 2),
                    'Std_ms': round(s['std'], 2),
                    'P95_ms': round(s['p95'], 2),
                    'Min_ms': round(s['min'], 2),
                    'Max_ms': round(s['max'], 2),
                    'Samples': s['count'],
                })
        return pd.DataFrame(rows)

    def to_thesis_table(self):
        stats = self.get_statistics()
        rows = []
        for stage in self.stage_names:
            if stats[stage]:
                s = stats[stage]
                name = stage.replace('_', ' ').title()
                if stage == 'yolo_inference':
                    name = 'YOLO Inference (640×640)'
                elif stage == 'pose_inference':
                    name = 'Pose Estimation'
                elif stage == 'total_pipeline':
                    name = '**Total Pipeline**'
                rows.append({
                    'Component': name,
                    'Mean (ms)': f"{s['mean']:.1f}",
                    '95th Percentile (ms)': f"{s['p95']:.1f}",
                })
        return pd.DataFrame(rows)


# =============================================================================
# YOLO UTILITIES
# =============================================================================

def load_yolo_model(model_path: str):
    """Load YOLO model — tries Ultralytics (v8/v11) first, falls back to YOLOv5 torch.hub."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        st.info("✅ Loaded with Ultralytics (YOLOv8+)")
        return model, 'v8'
    except Exception as e:
        if "YOLOv5" in str(e) or "forwards compatible" in str(e):
            try:
                st.warning("⚠️ Detected YOLOv5 model — loading with torch.hub...")
                model = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path=model_path, force_reload=False)
                st.info("✅ Loaded YOLOv5 model")
                return model, 'v5'
            except Exception as e2:
                st.error(f"Failed to load YOLOv5: {e2}")
                return None, None
        else:
            st.error(f"Failed to load YOLO: {e}")
            return None, None


def load_classes_from_yaml(yaml_path: str) -> List[str]:
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    except Exception:
        return []


def boxes_intersect(boxA, boxB):
    return not (boxA[2] < boxB[0] or boxB[2] < boxA[0] or
                boxA[3] < boxB[1] or boxB[3] < boxA[1])


def merge_overlapping_boxes(boxes: List) -> List:
    """Merge overlapping bounding boxes (iterative union)."""
    if not boxes:
        return []
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        placed = False
        for i, (mx1, my1, mx2, my2) in enumerate(merged):
            if boxes_intersect((x1, y1, x2, y2), (mx1, my1, mx2, my2)):
                merged[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                placed = True
                break
        if not placed:
            merged.append((int(x1), int(y1), int(x2), int(y2)))
    # Second pass
    changed = True
    while changed:
        changed = False
        new_merged = []
        for box in merged:
            placed = False
            for j, nbox in enumerate(new_merged):
                if boxes_intersect(box, nbox):
                    new_merged[j] = (min(box[0], nbox[0]), min(box[1], nbox[1]),
                                     max(box[2], nbox[2]), max(box[3], nbox[3]))
                    placed = True
                    changed = True
                    break
            if not placed:
                new_merged.append(box)
        merged = new_merged
    return merged


def yolo_inference_v5(frame, model):
    """YOLOv5 inference via torch.hub"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model.iou = 0.2
    model.conf = 0.1
    results = model(rgb)
    try:
        return results.xyxy[0].cpu().numpy()
    except Exception:
        return np.array([])


def yolo_inference_v8(frame, model):
    """YOLOv8/v11 inference via Ultralytics"""
    results = model(frame, verbose=False)
    if results and len(results[0].boxes) > 0:
        return results[0].boxes.data.cpu().numpy()
    return np.array([])


# =============================================================================
# VIDEO PROCESSING PIPELINE
# =============================================================================

def process_video(video_path: str,
                  yolo_model, yolo_type: str,
                  class_names: List[str],
                  enable_yolo: bool = True,
                  enable_pose: bool = True,
                  pose_params: Dict = None,
                  enable_benchmark: bool = True,
                  stream_output: bool = True) -> Tuple[List[Dict], str, Dict]:
    """
    Process video with YOLO detection + Pose analytics.
    
    Returns:
        (frame_data, output_video_path, summary_metrics)
    """
    # Setup
    os.makedirs("runs", exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"runs/{base}_analyzed.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return [], "", {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Initialize pose analytics
    pose_engine = None
    if enable_pose:
        with st.spinner("Initializing pose detector..."):
            params = pose_params or {}
            pose_engine = PoseAnalytics(
                history_length=params.get('history_length', 7),
                model_complexity=params.get('model_complexity', 1),
                min_detection_confidence=params.get('min_detection_confidence', 0.5),
                min_tracking_confidence=params.get('min_tracking_confidence', 0.5),
            )
            pose_engine.initialize()

    # Benchmark
    benchmark = PipelineBenchmark() if enable_benchmark else None
    if benchmark:
        benchmark.capture_hardware_info()

    # GPU warmup
    if enable_benchmark and torch.cuda.is_available() and enable_yolo and yolo_model:
        st.info("🔥 Warming up GPU for accurate benchmarking...")
        ret, warmup = cap.read()
        if ret:
            infer = yolo_inference_v8 if yolo_type == 'v8' else yolo_inference_v5
            for _ in range(10):
                infer(warmup, yolo_model)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            torch.cuda.synchronize()

    # UI placeholders
    if stream_output:
        st.subheader("📹 Processing Output")
        col_vid, col_stats = st.columns([3, 1])
        with col_vid:
            frame_placeholder = st.empty()
        with col_stats:
            stats_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

    # YOLO event tracking
    CONF_THRESH = 0.25
    counters = {'punch': 0, 'kick': 0}
    in_event = {'punch': False, 'kick': False}
    event_start = {'punch': 0, 'kick': 0}
    gap_counter = {'punch': 0, 'kick': 0}
    gap_tolerance = {'punch': 2, 'kick': 4}
    min_event_dur = {'punch': 2, 'kick': 3}

    # Frame data collection
    all_frame_data = []
    velocity_history_chart = {
        'frames': deque(maxlen=120),
        'wrist': deque(maxlen=120),
        'ankle': deque(maxlen=120),
    }

    frame_idx = 0
    prev_time = time.time()

    st.info(f"🎬 Processing {total_frames} frames at {fps:.1f} fps...")

    while True:
        # STAGE 1: Frame extraction
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            break
        if benchmark:
            benchmark.record('frame_extraction', (t1 - t0) * 1000)

        t_pipeline = time.perf_counter()

        # FPS tracking
        now = time.time()
        elapsed = now - prev_time
        fps_display = 1.0 / elapsed if elapsed > 0 else 0
        prev_time = now

        # STAGE 2: YOLO inference
        ov_punch = False
        ov_kick = False
        bag_boxes = []
        filtered_dets = []

        if enable_yolo and yolo_model:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_yolo_start = time.perf_counter()

            infer = yolo_inference_v8 if yolo_type == 'v8' else yolo_inference_v5
            detections = infer(frame, yolo_model)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_yolo_end = time.perf_counter()
            if benchmark:
                benchmark.record('yolo_inference', (t_yolo_end - t_yolo_start) * 1000)

            # Filter detections
            filtered_dets = [d for d in detections if d[4] >= CONF_THRESH]

            raw_bag = [tuple(map(int, d[:4])) for d in filtered_dets if int(d[5]) == 0]
            raw_punch = [tuple(map(int, d[:4])) for d in filtered_dets if int(d[5]) == 5]
            raw_kick = [tuple(map(int, d[:4])) for d in filtered_dets if int(d[5]) == 2]

            bag_boxes = merge_overlapping_boxes(raw_bag)
            punch_boxes = merge_overlapping_boxes(raw_punch)
            kick_boxes = merge_overlapping_boxes(raw_kick)

            # STAGE 3: Hit detection
            t_hit_start = time.perf_counter()

            # Check punch-bag overlap
            for a_box in punch_boxes:
                for b_box in bag_boxes:
                    if boxes_intersect(a_box, b_box):
                        ov_punch = True
                        break
                if ov_punch:
                    break

            # Check kick-bag overlap
            for a_box in kick_boxes:
                for b_box in bag_boxes:
                    if boxes_intersect(a_box, b_box):
                        ov_kick = True
                        break
                if ov_kick:
                    break

            # Event counting (debounced)
            for action, is_hit in [('punch', ov_punch), ('kick', ov_kick)]:
                if is_hit:
                    gap_counter[action] = 0
                    if not in_event[action]:
                        in_event[action] = True
                        event_start[action] = frame_idx
                else:
                    if in_event[action]:
                        gap_counter[action] += 1
                        if gap_counter[action] >= gap_tolerance[action]:
                            dur = frame_idx - event_start[action]
                            if dur >= min_event_dur[action]:
                                counters[action] += 1
                            in_event[action] = False
                            gap_counter[action] = 0

            t_hit_end = time.perf_counter()
            if benchmark:
                benchmark.record('hit_detection', (t_hit_end - t_hit_start) * 1000)

        # STAGE 3b: Pose inference
        pose_result = None
        if enable_pose and pose_engine:
            t_pose_start = time.perf_counter()
            pose_result = pose_engine.process_frame(
                frame, frame_idx,
                yolo_punch_hit=ov_punch,
                yolo_kick_hit=ov_kick,
            )
            t_pose_end = time.perf_counter()
            if benchmark:
                benchmark.record('pose_inference', (t_pose_end - t_pose_start) * 1000)

        # STAGE 4: Overlay rendering
        t_overlay_start = time.perf_counter()

        # Draw YOLO boxes
        if enable_yolo:
            for det in filtered_dets:
                x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
                cls_id = int(cls_id)
                color = {
                    0: (0, 255, 0), 5: (255, 0, 255), 2: (0, 255, 255),
                    4: (255, 0, 0), 1: (0, 128, 255), 3: (255, 165, 0),
                }.get(cls_id, (128, 128, 128))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                if cls_id < len(class_names):
                    cv2.putText(frame, f"{class_names[cls_id]} {conf:.2f}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw overlap lines
            if ov_punch:
                for a in punch_boxes:
                    for b in bag_boxes:
                        if boxes_intersect(a, b):
                            ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                            cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                            cv2.line(frame, ca, cb, (0, 255, 255), 2)
            if ov_kick:
                for a in kick_boxes:
                    for b in bag_boxes:
                        if boxes_intersect(a, b):
                            ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                            cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                            cv2.line(frame, ca, cb, (0, 255, 255), 2)

        # Draw pose overlay
        if pose_result and pose_result.get('pose_detected'):
            frame = pose_engine.draw_overlay(frame, pose_result,
                                             draw_skeleton=True,
                                             draw_velocity=True)

        # Stats overlay on frame
        cv2.rectangle(frame, (5, 5), (280, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (280, 160), (255, 255, 255), 1)
        y = 28
        cv2.putText(frame, f"Punches: {counters['punch']}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        y += 30
        cv2.putText(frame, f"Kicks: {counters['kick']}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Total: {counters['punch'] + counters['kick']}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if pose_result and pose_result.get('velocities'):
            vel = pose_result['velocities']
            max_w = max(vel.get('left_wrist', 0), vel.get('right_wrist', 0))
            max_a = max(vel.get('left_ankle', 0), vel.get('right_ankle', 0))
            y += 30
            cv2.putText(frame, f"Wrist Vel: {max_w:.3f}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 200), 1)
            y += 22
            cv2.putText(frame, f"Ankle Vel: {max_a:.3f}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

        # FPS
        fps_text = f"FPS: {fps_display:.1f}"
        ts = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, fps_text, (w - ts[0] - 10, ts[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

        t_overlay_end = time.perf_counter()
        if benchmark:
            benchmark.record('overlay_rendering', (t_overlay_end - t_overlay_start) * 1000)
            benchmark.record('total_pipeline', (time.perf_counter() - t_pipeline) * 1000)

        # Collect frame data
        frame_info = {
            'frame': frame_idx,
            'timestamp': frame_idx / fps,
            'punch_hit': ov_punch,
            'kick_hit': ov_kick,
            'punch_count': counters['punch'],
            'kick_count': counters['kick'],
        }

        if pose_result:
            vel = pose_result.get('velocities', {})
            ext = pose_result.get('extensions', {})
            guard = pose_result.get('guard', {})
            frame_info.update({
                'wrist_vel_max': max(vel.get('left_wrist', 0), vel.get('right_wrist', 0)),
                'ankle_vel_max': max(vel.get('left_ankle', 0), vel.get('right_ankle', 0)),
                'knee_vel_max': max(vel.get('left_knee', 0), vel.get('right_knee', 0)),
                'left_arm_ext': ext.get('left_arm', 0),
                'right_arm_ext': ext.get('right_arm', 0),
                'left_leg_ext': ext.get('left_leg', 0),
                'right_leg_ext': ext.get('right_leg', 0),
                'left_guard': guard.get('left_guard'),
                'right_guard': guard.get('right_guard'),
            })

            velocity_history_chart['frames'].append(frame_idx)
            velocity_history_chart['wrist'].append(frame_info['wrist_vel_max'])
            velocity_history_chart['ankle'].append(frame_info['ankle_vel_max'])

        all_frame_data.append(frame_info)

        # Stream to UI (every 5 frames)
        if stream_output and frame_idx % 5 == 0:
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display, channels="RGB", use_container_width=True)

            stats_md = f"""
**Frame:** {frame_idx}/{total_frames}

**Strikes:**
- Punches: {counters['punch']}
- Kicks: {counters['kick']}
- Total: {counters['punch'] + counters['kick']}
"""
            stats_placeholder.markdown(stats_md)

            progress = min(frame_idx / max(total_frames, 1), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_idx}/{total_frames} ({progress*100:.0f}%)")

        frame_idx += 1

    # Finalize in-progress events
    for action in ('punch', 'kick'):
        if in_event[action]:
            dur = frame_idx - event_start[action]
            if dur >= min_event_dur[action]:
                counters[action] += 1

    cap.release()
    out.release()

    # Get summary metrics
    summary_metrics = {}
    if pose_engine:
        summary_metrics = pose_engine.get_summary_metrics()
        pose_engine.close()

    if stream_output:
        progress_bar.empty()
        status_text.empty()

    # Save frame-level CSV
    df = pd.DataFrame(all_frame_data)
    csv_path = f"runs/{base}_frame_data.csv"
    df.to_csv(csv_path, index=False)

    # Save ground truth
    ground_truth = {
        'video': base, 'total_frames': frame_idx,
        'punch_count': counters['punch'], 'kick_count': counters['kick'],
        'fps': fps,
    }
    gt_path = f"runs/{base}_ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    st.success(f"""
✅ **Processing Complete!**

**Final Counts:** Punches: {counters['punch']} | Kicks: {counters['kick']} | Total: {counters['punch'] + counters['kick']}

📁 Frame data: `{csv_path}` | 🎥 Video: `{out_path}`
""")

    # Benchmark display
    if benchmark:
        display_benchmark(benchmark, w, h, fps, frame_idx, base)

    return all_frame_data, out_path, summary_metrics


# =============================================================================
# BENCHMARK DISPLAY
# =============================================================================

def display_benchmark(benchmark: PipelineBenchmark, width, height, fps, frames, base):
    st.divider()
    st.subheader("📊 Pipeline Latency Benchmark (Thesis Table 4.2)")

    thesis_df = benchmark.to_thesis_table()
    st.table(thesis_df)

    full_df = benchmark.to_dataframe()
    st.dataframe(full_df)

    bench_csv = f"runs/{base}_benchmark.csv"
    full_df.to_csv(bench_csv, index=False)
    st.success(f"📁 Benchmark data saved: `{bench_csv}`")

    stats = benchmark.get_statistics()
    total_mean = stats['total_pipeline']['mean'] if stats['total_pipeline'] else 0
    total_p95 = stats['total_pipeline']['p95'] if stats['total_pipeline'] else 0

    st.subheader("⏱️ Real-Time Capability")
    fps_thresh = 33.3
    if total_mean < fps_thresh:
        st.success(f"✅ Real-time capable: {total_mean:.1f} ms mean < 33.3 ms (30 FPS)")
    else:
        st.warning(f"⚠️ Mean latency {total_mean:.1f} ms exceeds 33.3 ms threshold")

    achievable_fps = 1000 / total_mean if total_mean > 0 else 0
    st.write(f"Achievable FPS (mean): **{achievable_fps:.1f}** | "
             f"Achievable FPS (P95): **{1000 / total_p95:.1f}**" if total_p95 > 0 else "")

    hw = benchmark.hardware_info
    st.subheader("🖥️ Hardware Configuration")
    hw_text = f"Device: {hw.get('device')} | CUDA: {hw.get('cuda_available')}"
    if hw.get('cuda_available'):
        hw_text += f" | GPU: {hw.get('gpu_name')} | CUDA: {hw.get('cuda_version')}"
    hw_text += f" | PyTorch: {hw.get('pytorch_version')} | Resolution: {width}×{height} | FPS: {fps}"
    st.write(hw_text)


# =============================================================================
# ANALYTICS VISUALIZATION
# =============================================================================

def display_analytics(frame_data: List[Dict], metrics: Dict):
    """Display comprehensive analytics dashboard."""
    st.header("📊 Training Analytics")

    df = pd.DataFrame(frame_data)

    # Summary row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Frames", len(df))
    with col2:
        st.metric("Duration", f"{df['timestamp'].max():.1f}s")
    with col3:
        st.metric("Punches", df['punch_count'].max())
    with col4:
        st.metric("Kicks", df['kick_count'].max())
    with col5:
        total = df['punch_count'].max() + df['kick_count'].max()
        st.metric("Total Strikes", total)

    st.divider()

    # Velocity analysis
    if 'wrist_vel_max' in df.columns:
        st.subheader("🚀 Velocity Analysis")
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['wrist_vel_max'],
                mode='lines', name='Wrist (Punches)',
                line=dict(color='rgb(255,100,255)', width=1),
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['ankle_vel_max'],
                mode='lines', name='Ankle (Kicks)',
                line=dict(color='rgb(100,255,255)', width=1),
            ))

            # Mark YOLO-detected strikes
            punches = df[df['punch_hit'] == True]
            if len(punches) > 0:
                fig.add_trace(go.Scatter(
                    x=punches['timestamp'], y=punches['wrist_vel_max'],
                    mode='markers', name='Punch Impact',
                    marker=dict(color='red', size=10, symbol='star'),
                ))
            kicks = df[df['kick_hit'] == True]
            if len(kicks) > 0:
                fig.add_trace(go.Scatter(
                    x=kicks['timestamp'], y=kicks['ankle_vel_max'],
                    mode='markers', name='Kick Impact',
                    marker=dict(color='orange', size=10, symbol='star'),
                ))

            fig.update_layout(
                title="Joint Velocity Over Time (strikes marked ★)",
                xaxis_title="Time (s)", yaxis_title="Velocity (norm units/frame)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Wrist Velocity Distribution",
                                                "Ankle Velocity Distribution"),
                                vertical_spacing=0.15)
            fig.add_trace(go.Histogram(
                x=df['wrist_vel_max'], nbinsx=50, name='Wrist',
                marker_color='rgb(255,100,255)'), row=1, col=1)
            fig.add_trace(go.Histogram(
                x=df['ankle_vel_max'], nbinsx=50, name='Ankle',
                marker_color='rgb(100,255,255)'), row=2, col=1)
            fig.update_layout(height=400, showlegend=False, title_text="Velocity Distributions")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Extension analysis
    if 'left_arm_ext' in df.columns:
        st.subheader("💪 Limb Extension (Technique)")
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['left_arm_ext'],
                mode='lines', name='Left Arm', line=dict(width=1)))
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['right_arm_ext'],
                mode='lines', name='Right Arm', line=dict(width=1)))
            fig.update_layout(title="Arm Extension Over Time (1=straight, 0=bent)",
                              xaxis_title="Time (s)", yaxis_title="Extension",
                              height=350, yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['left_leg_ext'],
                mode='lines', name='Left Leg', line=dict(width=1)))
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['right_leg_ext'],
                mode='lines', name='Right Leg', line=dict(width=1)))
            fig.update_layout(title="Leg Extension Over Time",
                              xaxis_title="Time (s)", yaxis_title="Extension",
                              height=350, yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

    # Guard analysis
    if 'left_guard' in df.columns:
        st.subheader("🛡️ Guard Position")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['left_guard'],
            mode='lines', name='Left Hand', line=dict(color='rgb(100,200,255)', width=1)))
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['right_guard'],
            mode='lines', name='Right Hand', line=dict(color='rgb(255,200,100)', width=1)))
        fig.add_hline(y=0.6, line_dash="dash", line_color="green",
                      annotation_text="High Guard Threshold")
        fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                      annotation_text="Low Guard Warning")
        fig.update_layout(title="Guard Height (1=shoulder, 0=hip)",
                          xaxis_title="Time (s)", yaxis_title="Guard Height",
                          height=350, yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cumulative strikes
    st.subheader("⏱️ Strike Timeline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['punch_count'],
        mode='lines', name='Punches',
        line=dict(color='rgb(255,0,255)', width=2)))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['kick_count'],
        mode='lines', name='Kicks',
        line=dict(color='rgb(0,255,255)', width=2)))
    fig.update_layout(title="Cumulative Strike Count",
                      xaxis_title="Time (s)", yaxis_title="Count", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Impact biomechanics
    if metrics.get('strike_analytics') or metrics.get('punch_impact_vel_mean'):
        st.divider()
        st.subheader("📈 Strike Biomechanics (at YOLO-detected impacts)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Punch Impact**")
            if metrics.get('punch_impact_vel_mean') is not None:
                st.write(f"Mean velocity: {metrics['punch_impact_vel_mean']:.4f}")
                st.write(f"Max velocity: {metrics['punch_impact_vel_max']:.4f}")
                if metrics.get('punch_ext_mean') is not None:
                    st.write(f"Mean arm extension: {metrics['punch_ext_mean']:.2f}")
            else:
                st.write("No punch impacts recorded")

        with col2:
            st.markdown("**Kick Impact**")
            if metrics.get('kick_impact_vel_mean') is not None:
                st.write(f"Mean velocity: {metrics['kick_impact_vel_mean']:.4f}")
                st.write(f"Max velocity: {metrics['kick_impact_vel_max']:.4f}")
                if metrics.get('kick_ext_mean') is not None:
                    st.write(f"Mean leg extension: {metrics['kick_ext_mean']:.2f}")
            else:
                st.write("No kick impacts recorded")

        with col3:
            st.markdown("**Guard Analysis**")
            for side in ['left', 'right']:
                mean = metrics.get(f'{side}_guard_mean')
                high = metrics.get(f'{side}_guard_high_pct')
                if mean is not None:
                    st.write(f"{side.title()} guard: {mean:.2f} avg, {high:.0f}% high")


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.title("🥊 Combat Sports Analysis")
    st.markdown("**YOLO** detection + **Pose** biomechanics for combat sports training analysis.")

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # YOLO model selection
    yolo_models = []
    if os.path.exists('models'):
        yolo_models = [f for f in os.listdir('models') if f.endswith('.pt')]

    model_path = None
    if yolo_models:
        selected = st.sidebar.selectbox("YOLO Model", yolo_models)
        model_path = os.path.join('models', selected)
    else:
        st.sidebar.warning("⚠️ No YOLO models found in `models/` folder")

    # Class names
    yaml_path = os.path.join("dataset", "data.yaml")
    class_names = load_classes_from_yaml(yaml_path) if os.path.exists(yaml_path) else \
                  ['bag', 'high-guard', 'kick-knee', 'low-guard', 'person', 'punch']

    # Detection toggles
    st.sidebar.subheader("Detection Modules")
    enable_yolo = st.sidebar.checkbox("✅ YOLO Object Detection", value=True)
    enable_pose = st.sidebar.checkbox("✅ Pose Biomechanics", value=True)
    enable_benchmark = st.sidebar.checkbox("⏱️ Pipeline Benchmarking", value=True,
                                           help="Collect latency data for Thesis Table 4.2")
    stream_output = st.sidebar.checkbox("📺 Stream Output Video", value=True)

    # Pose parameters
    with st.sidebar.expander("🎯 Pose Settings", expanded=False):
        model_complexity = st.selectbox("Model Complexity",
                                        [0, 1, 2],
                                        index=1,
                                        format_func=lambda x: {0: "Lite (fastest)",
                                                                1: "Full (balanced)",
                                                                2: "Heavy (most accurate)"}[x])
        det_conf = st.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05)
        track_conf = st.slider("Tracking Confidence", 0.1, 0.9, 0.5, 0.05)
        hist_len = st.slider("Velocity History (frames)", 3, 15, 7)

    pose_params = {
        'model_complexity': model_complexity,
        'min_detection_confidence': det_conf,
        'min_tracking_confidence': track_conf,
        'history_length': hist_len,
    }

    # Main content
    st.header("📤 Upload Video")
    video_file = st.file_uploader("Select training video", type=['mp4', 'avi', 'mov'])

    if video_file:
        os.makedirs("data", exist_ok=True)
        video_path = os.path.join("data", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(video_file)
        with col2:
            st.info(f"""
**Video Info:**
- Name: {video_file.name}
- Size: {video_file.size / 1024 / 1024:.1f} MB

Click **Run Analysis** to process.
""")

        if st.button("🚀 Run Analysis", type="primary"):
            yolo_model = None
            yolo_type = None

            if enable_yolo and model_path:
                with st.spinner("Loading YOLO model..."):
                    yolo_model, yolo_type = load_yolo_model(model_path)
                    if yolo_model and yolo_type == 'v8':
                        yolo_model.to(device)

            frame_data, out_path, metrics = process_video(
                video_path, yolo_model, yolo_type, class_names,
                enable_yolo=enable_yolo,
                enable_pose=enable_pose,
                pose_params=pose_params,
                enable_benchmark=enable_benchmark,
                stream_output=stream_output,
            )

            if frame_data:
                display_analytics(frame_data, metrics)

            # Download buttons
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if out_path and os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.download_button("📥 Download Processed Video",
                                           f.read(), os.path.basename(out_path),
                                           "video/mp4", use_container_width=True)
            with col2:
                csv_path = out_path.replace('_analyzed.mp4', '_frame_data.csv')
                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button("📊 Download CSV Data",
                                           f.read(), os.path.basename(csv_path),
                                           "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
