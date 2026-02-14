# Important Imports
import os
import subprocess
import json
import sys
import socket
import pathlib
import logging
import warnings
import time
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import streamlit as st
import cv2
import shutil
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# Error Handling
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
torch.classes.__path__ = []   # neutralize the broken proxy
warnings.filterwarnings("ignore", category=FutureWarning)

# Check if Device is CUDA ready
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Avoid Logging Issues
logging.getLogger("torch").setLevel(logging.ERROR)
st.set_page_config(page_title="Combat Sports Prototype", layout="wide")
pathlib.PosixPath = pathlib.WindowsPath

# Try import HMM learn module
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

# Session state setups for important variables
if "dataset_folder" not in st.session_state:
    st.session_state["dataset_folder"] = None
if "data_yaml_path" not in st.session_state:
    st.session_state["data_yaml_path"] = None
if "yolo_classes" not in st.session_state:
    st.session_state["yolo_classes"] = []
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

# =============================================================================
# BENCHMARKING CLASS - For measuring pipeline latencies (Thesis Table 4.2)
# =============================================================================
class PipelineBenchmark:
    """
    Collects timing measurements for each pipeline stage.
    Use this to generate Table 4.2 data for your thesis.
    
    Hardware tested on:
    - GPU: NVIDIA GeForce RTX 4060 Laptop GPU
    - CPU: Intel Core Ultra 7 155H (3.80 GHz)
    - RAM: 32.0 GB
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self.stage_names = [
            'frame_extraction',
            'yolo_inference', 
            'hit_detection',
            'overlay_rendering',
            'total_pipeline'
        ]
        self.hardware_info = {}
    
    def record(self, stage_name, duration_ms):
        """Record a timing measurement in milliseconds"""
        self.timings[stage_name].append(duration_ms)
    
    def capture_hardware_info(self):
        """Capture hardware configuration for thesis documentation"""
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
        """Calculate mean and 95th percentile for each stage"""
        stats = {}
        for stage in self.stage_names:
            if self.timings[stage]:
                arr = np.array(self.timings[stage])
                stats[stage] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'median': np.median(arr),
                    'p95': np.percentile(arr, 95),
                    'p99': np.percentile(arr, 99),
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'count': len(arr)
                }
            else:
                stats[stage] = None
        return stats
    
    def print_report(self):
        """Print a formatted report suitable for thesis Table 4.2"""
        stats = self.get_statistics()
        print("\n" + "="*70)
        print("PIPELINE LATENCY BENCHMARK RESULTS - FOR THESIS TABLE 4.2")
        print("="*70)
        print(f"{'Component':<35} {'Mean (ms)':<12} {'95th % (ms)':<12}")
        print("-"*70)
        
        for stage in self.stage_names:
            if stats[stage]:
                s = stats[stage]
                display_name = stage.replace('_', ' ').title()
                if stage == 'yolo_inference':
                    display_name = 'YOLO inference (640×640)'
                print(f"{display_name:<35} {s['mean']:.2f}         {s['p95']:.2f}")
        
        print("-"*70)
        if stats['frame_extraction']:
            print(f"Total frames processed: {stats['frame_extraction']['count']}")
        print("="*70)
        return stats
    
    def to_dataframe(self):
        """Export results as DataFrame for easy CSV export"""
        stats = self.get_statistics()
        rows = []
        for stage in self.stage_names:
            if stats[stage]:
                s = stats[stage]
                display_name = stage.replace('_', ' ').title()
                if stage == 'yolo_inference':
                    display_name = 'YOLO inference (640×640)'
                rows.append({
                    'Component': display_name,
                    'Mean_ms': round(s['mean'], 2),
                    'Std_ms': round(s['std'], 2),
                    'Median_ms': round(s['median'], 2),
                    'P95_ms': round(s['p95'], 2),
                    'P99_ms': round(s['p99'], 2),
                    'Min_ms': round(s['min'], 2),
                    'Max_ms': round(s['max'], 2),
                    'Sample_Count': s['count']
                })
        return pd.DataFrame(rows)
    
    def to_thesis_table(self):
        """Generate a table formatted exactly for thesis Table 4.2"""
        stats = self.get_statistics()
        rows = []
        for stage in self.stage_names:
            if stats[stage]:
                s = stats[stage]
                display_name = stage.replace('_', ' ').title()
                if stage == 'yolo_inference':
                    display_name = 'YOLO inference (640×640)'
                elif stage == 'total_pipeline':
                    display_name = '**Total pipeline**'
                rows.append({
                    'Component': display_name,
                    'Mean (ms)': f"{s['mean']:.1f}" if stage != 'total_pipeline' else f"**{s['mean']:.1f}**",
                    '95th Percentile (ms)': f"{s['p95']:.1f}" if stage != 'total_pipeline' else f"**{s['p95']:.1f}**"
                })
        return pd.DataFrame(rows)

# Global benchmark instance
pipeline_benchmark = PipelineBenchmark()

### Helper Functions
def check_internet_connection():
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except OSError:
        return False

def data_collection():
    st.title("Data Collection")
    uploaded_files = st.file_uploader("Upload Videos/Images for Inference or Dataset Training.", accept_multiple_files=True, type=['mp4', 'avi', 'mov', 'jpg', 'png'])
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            valid_extensions = ['.mp4', '.avi', '.mov', '.jpg', '.png']
            if file_extension in valid_extensions:
                sanitized_filename = os.path.basename(uploaded_file.name)
                save_path = os.path.join(data_dir, sanitized_filename)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {sanitized_filename}")
    st.markdown(
        """
        ### Labeling Tools
        - [Roboflow](https://roboflow.com/)
        - [LabelImg](https://github.com/tzutalin/labelImg)
        - [CVAT](https://github.com/openvinotoolkit/cvat)
        """
    )

def load_classes_from_yaml(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8', errors='replace') as file:
            data = yaml.safe_load(file)
            class_names = data.get('names', [])
            if class_names:
                st.session_state['yolo_classes'] = class_names
            return class_names
    except:
        return []

def update_yaml_paths(yaml_path, dataset_folder):
    if not yaml_path or not dataset_folder:
        raise ValueError("Missing path values")
    with open(yaml_path, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data['train'] = os.path.join(dataset_folder, 'train/images').replace('\\', '/')
    yaml_data['val'] = os.path.join(dataset_folder, 'valid/images').replace('\\', '/')
    yaml_data['test'] = os.path.join(dataset_folder, 'test/images').replace('\\', '/')
    with open(yaml_path, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
    st.success("Updated YAML paths.")

@st.cache_resource
def load_yolo_model(weights='models/best.pt'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def yolo_training():
    st.title("YOLO Model Training")
    st.subheader("Select Dataset")
    default_yaml_path = os.path.join(os.getcwd(), "dataset", "data.yaml")
    st.write("Upload data.yaml or rely on default if exists.")
    user_file = st.file_uploader("Upload data.yaml", type="yaml")
    if user_file is not None:
        dataset_folder = os.path.join(os.getcwd(), "dataset")
        os.makedirs(dataset_folder, exist_ok=True)
        yaml_file_path = os.path.join(dataset_folder, "data.yaml")
        with open(yaml_file_path, "wb") as file:
            file.write(user_file.getbuffer())
        st.info(f"Uploaded file: {yaml_file_path}")
    else:
        if os.path.exists(default_yaml_path):
            dataset_folder = os.path.join(os.getcwd(), "dataset")
            yaml_file_path = default_yaml_path
        else:
            st.warning("No data.yaml found.")
            return
    try:
        update_yaml_paths(yaml_file_path, dataset_folder)
        st.session_state['dataset_folder'] = dataset_folder
        st.session_state['data_yaml_path'] = yaml_file_path
        yolo_classes = load_classes_from_yaml(yaml_file_path)
        if yolo_classes:
            st.success(f"Classes: {yolo_classes}")
    except:
        return
    dataset_path = st.session_state.get('dataset_folder')
    data_yaml_path = st.session_state.get('data_yaml_path')
    if dataset_path and data_yaml_path and os.path.exists(data_yaml_path):
        st.subheader("Training Params")
        epochs = st.number_input("Epochs", 1, 1000, 100)
        batch_size = st.number_input("Batch Size", 1, 64, 16)
        img_size = st.number_input("Image Size", 320, 1920, 640, step=32)
        yolo_model = st.selectbox("YOLO Model", ["yolov5s", "yolov5m", "yolov5l", "yolov5x"])
        if st.button("Start YOLO Training"):
            if not os.path.exists(dataset_path):
                st.error(f"No dataset: {dataset_path}")
                return
            if not os.path.exists(data_yaml_path):
                st.error(f"No YAML: {data_yaml_path}")
                return
            yolo_path = os.path.abspath("yolov5")
            train_py_path = os.path.join(yolo_path, "train.py")
            if not os.path.exists(train_py_path):
                st.error(f"No train.py: {train_py_path}")
                return
            command = [
                sys.executable, train_py_path,
                '--img', str(img_size),
                '--batch', str(batch_size),
                '--epochs', str(epochs),
                '--data', data_yaml_path,
                '--weights', f'{yolo_model}.pt',
                '--device', '0'
            ]
            progress_bar = st.progress(0)
            output_box = st.empty()
            try:
                with st.spinner('Training...'):
                    process = subprocess.Popen(command, cwd=yolo_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
                    total_lines = 0
                    for line in process.stdout:
                        if line.strip():
                            output_box.text(line.strip())
                        total_lines += 1
                        progress = min(total_lines / (epochs * 100), 1.0)
                        progress_bar.progress(int(progress * 100))
                    process.stdout.close()
                    process.wait()
                if process.returncode == 0:
                    st.success("YOLO training done.")
                    progress_bar.progress(100)
                else:
                    st.error("YOLO training failed.")
            except:
                st.error("Error occurred.")
    else:
        st.warning("No valid dataset or data.yaml")

def check_overlap(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    return not (x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1)

def boxes_intersect(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    if xA2 < xB1 or xB2 < xA1 or yA2 < yB1 or yB2 < yA1:
        return False
    return True

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = areaA + areaB - interArea
    return interArea / unionArea if unionArea > 0 else 0

def yolo_inference(frame, model):
    """Run YOLO inference on a single frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model.iou = 0.2
    model.conf = 0.1
    results = model(rgb_frame)
    try:
        detections = results.xyxy[0].cpu().numpy()
    except:
        return []
    return detections

def merge_overlapping_boxes(boxes):
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        placed = False
        for i, (mx1, my1, mx2, my2) in enumerate(merged):
            if boxes_intersect(box, (mx1, my1, mx2, my2)):
                merged[i] = (min(x1, mx1), min(y1, my1),
                             max(x2, mx2), max(y2, my2))
                placed = True
                break
        if not placed:
            merged.append(box)
    changed = True
    while changed:
        changed = False
        new_merged = []
        for box in merged:
            x1, y1, x2, y2 = box
            placed = False
            for j, (nx1, ny1, nx2, ny2) in enumerate(new_merged):
                if boxes_intersect(box, (nx1, ny1, nx2, ny2)):
                    new_merged[j] = (min(x1, nx1), min(y1, ny1),
                                     max(x2, nx2), max(y2, ny2))
                    placed = True
                    changed = True
                    break
            if not placed:
                new_merged.append(box)
        merged = new_merged
    return merged

def process_video(video_path, model, enable_benchmark=True):
    """
    Process video with optional benchmarking.
    Set enable_benchmark=True to collect timing data for thesis Table 4.2.
    
    NOW SAVES HIT INFORMATION TO CSV FOR TRANSFORMER TRAINING.
    """
    global pipeline_benchmark
    
    # Reset benchmark for new video
    if enable_benchmark:
        pipeline_benchmark = PipelineBenchmark()
        pipeline_benchmark.capture_hardware_info()
    
    class_names = st.session_state.get('yolo_classes', [])
    if not class_names:
        st.error("No class names available.")
        return [], ""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video: {video_path}")
        return [], ""

    os.makedirs("runs", exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"runs/{base}_processed.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    prev_frame_time = time.time()
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counters = {'punch': 0, 'kick-knee': 0}
    in_event = {'punch': False, 'kick-knee': False}
    event_start = {'punch': 0, 'kick-knee': 0}
    min_event_dur = {'punch': 2, 'kick-knee': 6}
    gap_counter = {'punch': 0, 'kick-knee': 0}
    gap_tolerance = {'punch': 1, 'kick-knee': 4}

    progress_bar = st.progress(0)
    status_text = st.empty()
    all_detections = []
    
    # NEW: Track frame-level data for transformer training
    frame_data = []  # Will store per-frame information including hits

    def check_overlap_local(action_boxes, bag_boxes, frame_img):
        for a_box in action_boxes:
            for b_box in bag_boxes:
                if boxes_intersect(a_box, b_box):
                    ca = ((a_box[0] + a_box[2]) // 2, (a_box[1] + a_box[3]) // 2)
                    cb = ((b_box[0] + b_box[2]) // 2, (b_box[1] + b_box[3]) // 2)
                    cv2.line(frame_img, ca, cb, (0, 255, 255), 2)
                    return True
        return False

    # GPU warmup - run a few inference passes to stabilize timing
    if enable_benchmark and torch.cuda.is_available():
        st.info("Warming up GPU for accurate benchmarking...")
        ret, warmup_frame = cap.read()
        if ret:
            for _ in range(10):
                _ = yolo_inference(warmup_frame, model)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        torch.cuda.synchronize()

    while True:
        # =====================================================================
        # STAGE 1: Frame Extraction - BENCHMARKED
        # =====================================================================
        t_frame_start = time.perf_counter()
        ret, frame = cap.read()
        t_frame_end = time.perf_counter()
        
        if not ret:
            break
        
        if enable_benchmark:
            pipeline_benchmark.record('frame_extraction', (t_frame_end - t_frame_start) * 1000)
        
        # Track total pipeline start (excluding frame read for fair comparison)
        t_pipeline_start = time.perf_counter()

        new_frame_time = time.time()
        elapsed = new_frame_time - prev_frame_time
        fps_display = 1.0 / elapsed if elapsed > 0 else 0
        prev_frame_time = new_frame_time

        # =====================================================================
        # STAGE 2: YOLO Inference - BENCHMARKED
        # =====================================================================
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure accurate GPU timing
        t_yolo_start = time.perf_counter()
        
        detections = yolo_inference(frame, model)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU to finish
        t_yolo_end = time.perf_counter()
        
        if enable_benchmark:
            pipeline_benchmark.record('yolo_inference', (t_yolo_end - t_yolo_start) * 1000)

        CONF_THRESH = 0.4
        filtered = [d for d in detections if d[4] >= CONF_THRESH]

        raw_bag = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 0]
        raw_punch = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 5]
        raw_kick = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 2]

        bag_boxes = merge_overlapping_boxes(raw_bag)
        punch_boxes = merge_overlapping_boxes(raw_punch)
        kick_boxes = merge_overlapping_boxes(raw_kick)

        # =====================================================================
        # STAGE 3: Hit Detection Computation - BENCHMARKED
        # =====================================================================
        t_hit_start = time.perf_counter()
        
        ov_punch = check_overlap_local(punch_boxes, bag_boxes, frame)
        ov_kick = check_overlap_local(kick_boxes, bag_boxes, frame)

        for action, is_over in [('punch', ov_punch), ('kick-knee', ov_kick)]:
            if is_over:
                gap_counter[action] = 0
                if not in_event[action]:
                    in_event[action] = True
                    event_start[action] = frame_count
            else:
                if in_event[action]:
                    gap_counter[action] += 1
                    if gap_counter[action] >= gap_tolerance[action]:
                        dur = frame_count - event_start[action]
                        if dur >= min_event_dur[action]:
                            counters[action] += 1
                        in_event[action] = False
                        gap_counter[action] = 0
        
        t_hit_end = time.perf_counter()
        
        if enable_benchmark:
            pipeline_benchmark.record('hit_detection', (t_hit_end - t_hit_start) * 1000)

        # =====================================================================
        # NEW: Collect frame-level data for transformer training
        # =====================================================================
        # Determine which classes are present in this frame
        classes_present = set()
        for d in filtered:
            cls_id = int(d[5])
            classes_present.add(cls_id)
        
        # Store frame data with hit information
        frame_info = {
            'frame': frame_count,
            'bag_present': 0 in classes_present,
            'high_guard_present': 1 in classes_present,
            'kick_present': 2 in classes_present,
            'low_guard_present': 3 in classes_present,
            'person_present': 4 in classes_present,
            'punch_present': 5 in classes_present,
            'punch_hit': ov_punch,  # CRITICAL: This is the actual hit detection
            'kick_hit': ov_kick,    # CRITICAL: This is the actual hit detection
            'in_punch_event': in_event['punch'],
            'in_kick_event': in_event['kick-knee'],
            'cumulative_punches': counters['punch'],
            'cumulative_kicks': counters['kick-knee']
        }
        frame_data.append(frame_info)

        # =====================================================================
        # STAGE 4: Overlay Rendering - BENCHMARKED
        # =====================================================================
        t_overlay_start = time.perf_counter()
        
        for det in filtered:
            x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
            cls_id = int(cls_id)
            color = {
                0: (0, 255, 0),    # bag
                5: (255, 0, 255),  # punch
                2: (0, 255, 255),  # kick
                4: (255, 0, 0),    # person
                1: (0, 128, 255),  # high-guard
                3: (255, 165, 0)   # low-guard
            }.get(cls_id, (0, 0, 255))
            
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        total_hits = counters['punch'] + counters['kick-knee']
        cv2.putText(frame, f"Hits: {total_hits}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Punch: {counters['punch']}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Kick-Knee: {counters['kick-knee']}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        fps_text = f"FPS: {fps_display:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x_pos = frame.shape[1] - text_size[0] - 10
        y_pos = text_size[1] + 10
        cv2.putText(frame, fps_text, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        
        t_overlay_end = time.perf_counter()
        
        if enable_benchmark:
            pipeline_benchmark.record('overlay_rendering', (t_overlay_end - t_overlay_start) * 1000)
        
        # =====================================================================
        # Total Pipeline Time
        # =====================================================================
        t_pipeline_end = time.perf_counter()
        
        if enable_benchmark:
            pipeline_benchmark.record('total_pipeline', (t_pipeline_end - t_pipeline_start) * 1000)

        all_detections.append(filtered)
        frame_count += 1
        
        if frame_count % 10 == 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress * 100:.1f}%)")

    for action in ('punch', 'kick-knee'):
        if in_event[action]:
            dur = frame_count - event_start[action]
            if dur >= min_event_dur[action]:
                counters[action] += 1

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    st.success(f"Processing complete!\nPunches: {counters['punch']}\nKicks: {counters['kick-knee']}")

    # =========================================================================
    # SAVE FRAME-LEVEL DATA FOR TRANSFORMER TRAINING
    # =========================================================================
    frame_df = pd.DataFrame(frame_data)
    frame_csv_path = f"runs/{base}_frame_data.csv"
    frame_df.to_csv(frame_csv_path, index=False)
    st.success(f"📁 Frame-level data saved to: `{frame_csv_path}` (use this for Transformer training)")
    
    # Also save the ground truth counts for validation
    ground_truth = {
        'video': base,
        'total_frames': frame_count,
        'yolo_punch_count': counters['punch'],
        'yolo_kick_count': counters['kick-knee'],
        'fps': fps
    }
    gt_path = f"runs/{base}_ground_truth.json"
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    st.info(f"Ground truth saved to: `{gt_path}`")

    # =========================================================================
    # DISPLAY BENCHMARK RESULTS FOR THESIS TABLE 4.2
    # =========================================================================
    if enable_benchmark:
        st.write("---")
        st.write("## 📊 Pipeline Latency Benchmark Results")
        st.write("**Use these values for thesis Table 4.2**")
        
        stats = pipeline_benchmark.get_statistics()
        
        # Create thesis-formatted table
        st.write("### Table 4.2: Measured Component Latencies")
        thesis_df = pipeline_benchmark.to_thesis_table()
        st.table(thesis_df)
        
        # Detailed statistics table
        st.write("### Detailed Statistics (for methodology section)")
        full_benchmark_df = pipeline_benchmark.to_dataframe()
        st.dataframe(full_benchmark_df)
        
        # Save to CSV
        benchmark_csv_path = f"runs/{base}_benchmark_results.csv"
        full_benchmark_df.to_csv(benchmark_csv_path, index=False)
        st.success(f"📁 Benchmark data saved to: `{benchmark_csv_path}`")
        
        # Print to console
        pipeline_benchmark.print_report()
        
        # Real-time capability assessment
        st.write("---")
        st.write("### ⏱️ Real-Time Capability Assessment")
        total_mean = stats['total_pipeline']['mean'] if stats['total_pipeline'] else 0
        total_p95 = stats['total_pipeline']['p95'] if stats['total_pipeline'] else 0
        fps_threshold = 33.3  # 30 FPS requirement
        
        if total_mean < fps_threshold:
            st.success(f"✅ **Real-time capable**: Mean latency ({total_mean:.1f} ms) < 33.3 ms threshold for 30 FPS")
        else:
            st.warning(f"⚠️ Mean latency ({total_mean:.1f} ms) exceeds 33.3 ms threshold - suitable for batch processing only")
        
        achievable_fps = 1000 / total_mean if total_mean > 0 else 0
        st.write(f"- **Achievable FPS (mean):** {achievable_fps:.1f}")
        st.write(f"- **Achievable FPS (95th percentile):** {1000 / total_p95:.1f}" if total_p95 > 0 else "")
        
        # Hardware info
        st.write("---")
        st.write("### 🖥️ Hardware Configuration (for thesis documentation)")
        hw = pipeline_benchmark.hardware_info
        st.write(f"- **Compute Device:** {hw.get('device', 'N/A')}")
        st.write(f"- **CUDA Available:** {hw.get('cuda_available', False)}")
        if hw.get('cuda_available'):
            st.write(f"- **GPU:** {hw.get('gpu_name', 'N/A')}")
            st.write(f"- **CUDA Version:** {hw.get('cuda_version', 'N/A')}")
        st.write(f"- **PyTorch Version:** {hw.get('pytorch_version', 'N/A')}")
        st.write(f"- **Video Resolution:** {width}×{height}")
        st.write(f"- **Video FPS:** {fps}")
        st.write(f"- **Total Frames Processed:** {frame_count}")
        
        # Copy-paste ready text for thesis
        st.write("---")
        st.write("### 📝 Copy-Paste Text for Thesis")
        thesis_text = f"""The measured total pipeline latency of {total_mean:.1f} ms (mean) is {'well below' if total_mean < fps_threshold else 'above'} the 33.3 ms threshold required for 30 FPS processing, {'confirming' if total_mean < fps_threshold else 'indicating limitations for'} real-time capability on the test hardware (NVIDIA RTX 4060 Laptop GPU). The 95th percentile latency of {total_p95:.1f} ms demonstrates consistent performance across frames."""
        st.code(thesis_text, language=None)

    return all_detections, out_path

def get_video_rotation(path: str) -> int:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "json",
            path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        info = json.loads(proc.stdout)
        return int(info["streams"][0]["tags"].get("rotate", 0))
    except Exception:
        return 0

def model_execution():
    st.title("Model Execution")
    
    yaml_file_path = os.path.join("dataset", "data.yaml")
    yolo_classes = load_classes_from_yaml(yaml_file_path)
    st.session_state['yolo_classes'] = yolo_classes

    yolo_models = [f for f in os.listdir('models') if f.endswith('.pt')]
    selected_yolo_model = st.selectbox("Select YOLO Model", yolo_models)
    model_path = os.path.join('models', selected_yolo_model)

    # Benchmark toggle
    st.write("---")
    enable_benchmark = st.checkbox("✅ Enable Latency Benchmarking (for Thesis Table 4.2)", value=True)
    if enable_benchmark:
        st.info("📊 Benchmarking enabled - timing data will be collected for each pipeline stage")

    video_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
    if video_file:
        video_path = os.path.join("data", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

    if st.button("Run Models") and video_file:
        model = load_yolo_model(model_path)
        if model is None:
            st.error("Failed to load YOLO model.")
            return
        model.to(device)

        detections, processed_video_path = process_video(video_path, model, enable_benchmark=enable_benchmark)

        if not detections:
            st.error("No detections.")
            return

        all_dets = []
        for frame_idx, frame in enumerate(detections):
            for d in frame:
                if len(d) >= 6:
                    class_id = int(d[5])
                    if 0 <= class_id < len(yolo_classes):
                        all_dets.append([frame_idx, *d[:6]])

        # Save legacy format for backwards compatibility
        csv_file_path = os.path.join("runs", f"yolo_predictions_{video_file.name}.csv")
        df = pd.DataFrame(all_dets, columns=['frame', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id'])
        df.to_csv(csv_file_path, index=False)
        st.success(f"Predictions saved to `{csv_file_path}`")

        if not os.path.exists(processed_video_path):
            st.error(f"Processed video not found:\n  {processed_video_path}")
            return

        with open(processed_video_path, "rb") as vid_f:
            st.download_button(
                label="Download Processed Video",
                data=vid_f.read(),
                file_name=os.path.basename(processed_video_path),
                mime="video/mp4"
            )


# =============================================================================
# TRANSFORMER FUNCTIONS - IMPROVED VERSION
# =============================================================================

class TransformerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ActionRecognitionTransformer(nn.Module):
    """
    Improved Transformer for action recognition.
    Now uses positional encoding and better architecture.
    """
    def __init__(self, input_size, d_model=64, nhead=2, num_layers=2, 
                 dim_feedforward=128, dropout=0.1, num_classes=2):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.1)
        
        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output head - predicts hit probability for each frame
        self.fc = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Embed input
        emb = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        emb = emb + self.pos_encoding[:, :seq_len, :]
        emb = self.dropout(emb)
        
        # Transformer encoding
        out = self.transformer_encoder(emb)
        out = self.layer_norm(out)
        
        # Output prediction for each frame
        out = self.fc(out)  # (batch, seq_len, num_classes)
        
        return out


def prepare_transformer_inputs_v2(csv_path, sequence_length=32, stride=8):
    """
    IMPROVED: Prepare transformer inputs using the new frame_data CSV format.
    This version uses actual hit labels from YOLO processing.
    
    Input features: [bag, high_guard, kick, low_guard, person, punch] presence (6 features)
    Output labels: [punch_hit, kick_hit] (2 classes)
    """
    df = pd.read_csv(csv_path)
    
    # Check for new format (frame_data CSV)
    required_cols = ['frame', 'punch_hit', 'kick_hit', 'bag_present', 'punch_present', 'kick_present']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must have columns: {required_cols}. Got: {list(df.columns)}")
    
    df = df.sort_values('frame').reset_index(drop=True)
    total_frames = len(df)
    
    if total_frames < sequence_length:
        raise ValueError(f"Not enough frames ({total_frames}) for sequence length ({sequence_length})")
    
    # Build feature matrix (6 features per frame)
    feature_cols = ['bag_present', 'high_guard_present', 'kick_present', 
                    'low_guard_present', 'person_present', 'punch_present']
    features = df[feature_cols].values.astype(np.float32)
    
    # Build label matrix (2 classes: punch_hit, kick_hit)
    label_cols = ['punch_hit', 'kick_hit']
    labels = df[label_cols].values.astype(np.float32)
    
    # Create sequences
    input_sequences = []
    label_sequences = []
    
    i = 0
    while i + sequence_length <= total_frames:
        seq_features = features[i:i + sequence_length]
        seq_labels = labels[i:i + sequence_length]
        
        input_sequences.append(torch.tensor(seq_features, dtype=torch.float32))
        label_sequences.append(torch.tensor(seq_labels, dtype=torch.float32))
        
        i += stride
    
    if not input_sequences:
        return torch.empty(0), torch.empty(0)
    
    inputs = torch.stack(input_sequences)
    labels_tensor = torch.stack(label_sequences)
    
    return inputs, labels_tensor


def prepare_transformer_inputs_legacy(csv_path, sequence_length=32, stride=8, num_classes=6):
    """
    Legacy function for old CSV format (yolo_predictions_*.csv).
    Converts to frame-level class presence vectors.
    
    NOTE: This version cannot detect actual hits - only class co-occurrence.
    For accurate hit detection, use the new frame_data CSV format.
    """
    df = pd.read_csv(csv_path)
    
    if 'frame' not in df.columns or 'class_id' not in df.columns:
        raise ValueError("CSV must have columns 'frame' and 'class_id'")
    
    df = df.sort_values('frame').reset_index(drop=True)
    
    if df.empty:
        return torch.empty(0), torch.empty(0)
    
    max_frame = int(df['frame'].max())
    
    # Build frame-level class presence vectors
    frame_vectors = np.zeros((max_frame + 1, num_classes), dtype=np.float32)
    
    for _, row in df.iterrows():
        frame_idx = int(row['frame'])
        class_id = int(row['class_id'])
        if 0 <= class_id < num_classes:
            frame_vectors[frame_idx, class_id] = 1.0
    
    # Create sequences
    input_sequences = []
    label_sequences = []
    
    i = 0
    while i + sequence_length <= max_frame + 1:
        seq = frame_vectors[i:i + sequence_length]
        input_sequences.append(torch.tensor(seq, dtype=torch.float32))
        
        # For legacy format, use class co-occurrence as proxy for hits
        # punch_hit proxy: punch (5) AND bag (0) both present
        # kick_hit proxy: kick (2) AND bag (0) both present
        hit_labels = np.zeros((sequence_length, 2), dtype=np.float32)
        for t in range(sequence_length):
            hit_labels[t, 0] = 1.0 if (seq[t, 5] > 0 and seq[t, 0] > 0) else 0.0  # punch hit
            hit_labels[t, 1] = 1.0 if (seq[t, 2] > 0 and seq[t, 0] > 0) else 0.0  # kick hit
        label_sequences.append(torch.tensor(hit_labels, dtype=torch.float32))
        
        i += stride
    
    if not input_sequences:
        return torch.empty(0), torch.empty(0)
    
    inputs = torch.stack(input_sequences)
    labels = torch.stack(label_sequences)
    
    return inputs, labels


def train_transformer_model_v2(inputs, labels, d_model=64, nhead=2, num_layers=2,
                                dim_feedforward=128, dropout=0.1, num_epochs=50,
                                batch_size=16, learning_rate=1e-3, model_save_path=None):
    """
    Improved transformer training with better hyperparameters and monitoring.
    """
    if len(inputs) == 0:
        st.error("No training data available")
        return None
    
    input_size = inputs.shape[-1]  # Number of input features
    num_classes = labels.shape[-1]  # Number of output classes
    
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Training on: {device_}")
    
    # Train/validation split
    N = len(inputs)
    split_idx = int(0.8 * N)
    
    if N < 5:
        st.warning("Very small dataset - using all data for training")
        train_in, train_lb = inputs, labels
        val_in, val_lb = inputs[:1], labels[:1]
    else:
        # Shuffle before splitting
        perm = torch.randperm(N)
        inputs = inputs[perm]
        labels = labels[perm]
        
        train_in = inputs[:split_idx]
        train_lb = labels[:split_idx]
        val_in = inputs[split_idx:]
        val_lb = labels[split_idx:]
    
    train_ds = TransformerDataset(train_in, train_lb)
    val_ds = TransformerDataset(val_in, val_lb)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model_ = ActionRecognitionTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes
    ).to(device_)
    
    # Loss function with class weighting for imbalanced data
    # Count positive samples to compute weights
    pos_counts = labels.sum(dim=(0, 1))
    total_samples = labels.shape[0] * labels.shape[1]
    pos_weight = (total_samples - pos_counts) / (pos_counts + 1e-6)
    pos_weight = torch.clamp(pos_weight, 1.0, 10.0).to(device_)
    
    st.info(f"Class weights (punch, kick): {pos_weight.cpu().numpy()}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model_.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_f1_scores = []
    
    temp_model_path = "models/transformer_model_temp.pth"
    
    for epoch in range(num_epochs):
        # Training phase
        model_.train()
        total_train_loss = 0.0
        
        for xb, yb in train_loader:
            xb = xb.to(device_)
            yb = yb.to(device_)
            
            optimizer.zero_grad()
            out = model_(xb)
            loss = criterion(out, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model_.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device_)
                yb = yb.to(device_)
                
                out = model_(xb)
                loss = criterion(out, yb)
                total_val_loss += loss.item()
                
                preds = torch.sigmoid(out) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        
        # Compute F1 score
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0).numpy().reshape(-1, num_classes)
            all_labels = torch.cat(all_labels, dim=0).numpy().reshape(-1, num_classes)
            
            # Per-class F1
            f1_scores = []
            for c in range(num_classes):
                tp = ((all_preds[:, c] == 1) & (all_labels[:, c] == 1)).sum()
                fp = ((all_preds[:, c] == 1) & (all_labels[:, c] == 0)).sum()
                fn = ((all_preds[:, c] == 0) & (all_labels[:, c] == 1)).sum()
                
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores)
            val_f1_scores.append(avg_f1)
        else:
            avg_f1 = 0.0
            val_f1_scores.append(0.0)
        
        # Update progress
        progress_bar.progress((epoch + 1) / num_epochs)
        status_text.text(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {avg_f1:.4f}")
        
        # Early stopping based on F1 score
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model_.state_dict(), temp_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                st.warning(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    if os.path.exists(temp_model_path):
        try:
            model_.load_state_dict(torch.load(temp_model_path, map_location=device_, weights_only=False))
        except TypeError:
            model_.load_state_dict(torch.load(temp_model_path, map_location=device_))
    
    # Save final model
    if model_save_path is None:
        model_save_path = "models/transformer_model.pth"
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model_.state_dict(),
        'input_size': input_size,
        'num_classes': num_classes,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'dropout': dropout,
        'best_val_f1': best_val_f1,
        'best_val_loss': best_val_loss
    }, model_save_path)
    
    st.success(f"✅ Model saved to: {model_save_path}")
    st.info(f"Best validation F1: {best_val_f1:.4f}, Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    
    axes[1].plot(val_f1_scores, label='Val F1', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score')
    axes[1].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return model_


def load_transformer_model_v2(model_path):
    """Load transformer model from checkpoint."""
    try:
        # Handle PyTorch 2.6+ weights_only default change
        try:
            # First try with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(model_path, map_location=device)
        
        # Handle both old and new checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            input_size = checkpoint.get('input_size', 6)
            num_classes = checkpoint.get('num_classes', 2)
            d_model = checkpoint.get('d_model', 64)
            nhead = checkpoint.get('nhead', 2)
            num_layers = checkpoint.get('num_layers', 2)
            dim_feedforward = checkpoint.get('dim_feedforward', 128)
            dropout = checkpoint.get('dropout', 0.1)
            
            model_ = ActionRecognitionTransformer(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_classes=num_classes
            ).to(device)
            
            model_.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Legacy format - try to load directly
            st.warning("Loading legacy model format - using default architecture")
            model_ = ActionRecognitionTransformer(
                input_size=6,
                d_model=64,
                nhead=2,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                num_classes=2
            ).to(device)
            model_.load_state_dict(checkpoint, strict=False)
        
        model_.eval()
        return model_
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def transformer_training_interface():
    st.title("Transformer Training (Hit Detection)")
    
    st.markdown("""
    ### Instructions
    1. First run **Model Execution** on a video to generate `*_frame_data.csv`
    2. Select that CSV file here for training
    3. The transformer will learn to predict punch/kick hits from class presence patterns
    """)
    
    # Find available CSVs
    frame_data_csvs = []
    legacy_csvs = []
    
    if os.path.exists('runs'):
        for f in os.listdir('runs'):
            if f.endswith('.csv'):
                if 'frame_data' in f:
                    frame_data_csvs.append(f)
                elif 'yolo_predictions' in f:
                    legacy_csvs.append(f)
    
    st.write("---")
    csv_type = st.radio(
        "Select CSV type:",
        ["Frame Data (Recommended)", "Legacy YOLO Predictions"],
        help="Frame Data CSVs contain actual hit labels. Legacy CSVs only have class presence."
    )
    
    if csv_type == "Frame Data (Recommended)":
        if not frame_data_csvs:
            st.warning("No frame_data CSVs found. Run Model Execution first to generate them.")
            return
        selected_csv = st.selectbox("Select Frame Data CSV", frame_data_csvs)
        csv_path = os.path.join("runs", selected_csv)
        use_legacy = False
    else:
        if not legacy_csvs:
            st.warning("No legacy prediction CSVs found.")
            return
        selected_csv = st.selectbox("Select Legacy CSV", legacy_csvs)
        csv_path = os.path.join("runs", selected_csv)
        use_legacy = True
        st.warning("⚠️ Legacy CSVs don't have true hit labels - using class co-occurrence as proxy")
    
    # Preview CSV
    st.write("### CSV Preview")
    df_preview = pd.read_csv(csv_path)
    st.write(df_preview.head(10))
    st.write(f"Total rows: {len(df_preview)}")
    
    if not use_legacy:
        # Show hit statistics
        punch_hits = df_preview['punch_hit'].sum()
        kick_hits = df_preview['kick_hit'].sum()
        st.write(f"**Punch hit frames:** {punch_hits} ({100*punch_hits/len(df_preview):.1f}%)")
        st.write(f"**Kick hit frames:** {kick_hits} ({100*kick_hits/len(df_preview):.1f}%)")
    
    st.write("---")
    st.write("### Training Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        sequence_length = st.number_input("Sequence Length", 16, 128, 32, step=8)
        stride = st.number_input("Stride", 1, 32, 8)
        num_epochs = st.number_input("Epochs", 10, 200, 50)
        batch_size = st.number_input("Batch Size", 4, 64, 16)
    
    with col2:
        d_model = st.number_input("Model Dimension", 32, 256, 64)
        nhead = st.selectbox("Attention Heads", [1, 2, 4, 8], index=1)
        num_layers = st.number_input("Transformer Layers", 1, 6, 2)
        learning_rate = st.select_slider("Learning Rate", 
                                          options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
                                          value=1e-3)
    
    model_name = st.text_input("Model Name", "transformer_hit_detector")
    
    if st.button("🚀 Start Training"):
        try:
            with st.spinner("Preparing data..."):
                if use_legacy:
                    inputs, labels = prepare_transformer_inputs_legacy(
                        csv_path, sequence_length=sequence_length, stride=stride
                    )
                else:
                    inputs, labels = prepare_transformer_inputs_v2(
                        csv_path, sequence_length=sequence_length, stride=stride
                    )
            
            st.write(f"**Input shape:** {inputs.shape}")
            st.write(f"**Labels shape:** {labels.shape}")
            
            if inputs.shape[0] < 10:
                st.error("Not enough training samples. Try reducing sequence length or stride.")
                return
            
            model_save_path = f"models/{model_name}.pth"
            
            model_ = train_transformer_model_v2(
                inputs, labels,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_save_path=model_save_path
            )
            
            if model_ is not None:
                st.success("✅ Training complete!")
                
        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# =============================================================================
# TRANSFORMER RESULTS DASHBOARD - FIXED VERSION
# =============================================================================

def transformer_results_dashboard():
    st.title("Transformer Results Dashboard")
    
    # Find transformer models
    if not os.path.exists("models"):
        st.warning("No models directory found.")
        return
    
    t_models = [f for f in os.listdir("models") if f.endswith(".pth")]
    if not t_models:
        st.warning("No transformer models found in models/")
        return
    
    selected_model = st.selectbox("Select Transformer Model", t_models)
    model_path = os.path.join("models", selected_model)
    
    # Find frame_data CSVs (exclude benchmark and other CSVs)
    if not os.path.exists("runs"):
        st.warning("No runs directory found.")
        return
    
    frame_data_csvs = [f for f in os.listdir("runs") 
                       if f.endswith('.csv') and 'frame_data' in f]
    legacy_csvs = [f for f in os.listdir("runs")
                   if f.endswith('.csv') and 'yolo_predictions' in f]
    
    st.write("---")
    csv_type = st.radio(
        "Select CSV type:",
        ["Frame Data (Recommended)", "Legacy YOLO Predictions"]
    )
    
    if csv_type == "Frame Data (Recommended)":
        if not frame_data_csvs:
            st.warning("No frame_data CSVs found. Run Model Execution first.")
            return
        selected_csv = st.selectbox("Select CSV for Analysis", frame_data_csvs)
        csv_path = os.path.join("runs", selected_csv)
        use_legacy = False
    else:
        if not legacy_csvs:
            st.warning("No legacy prediction CSVs found.")
            return
        selected_csv = st.selectbox("Select CSV for Analysis", legacy_csvs)
        csv_path = os.path.join("runs", selected_csv)
        use_legacy = True
    
    # Validate CSV before proceeding
    df = pd.read_csv(csv_path)
    
    if use_legacy:
        if 'frame' not in df.columns or 'class_id' not in df.columns:
            st.error(f"Invalid CSV format. Expected 'frame' and 'class_id' columns. Got: {list(df.columns)}")
            return
    else:
        required_cols = ['frame', 'punch_hit', 'kick_hit']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Invalid CSV format. Missing columns: {missing}")
            return
    
    st.write("### CSV Preview")
    st.write(df.head(10))
    
    # Inference parameters
    st.write("---")
    st.write("### Inference Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        sequence_length = st.number_input("Sequence Length", 16, 128, 32, step=8, key="inf_seq")
        stride = st.number_input("Stride", 1, 32, 4, key="inf_stride")
    with col2:
        punch_threshold = st.slider("Punch Detection Threshold", 0.1, 0.9, 0.5, 0.05)
        kick_threshold = st.slider("Kick Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    
    # Hit counting parameters (matching YOLO logic)
    st.write("### Hit Counting Parameters")
    col1, col2 = st.columns(2)
    with col1:
        punch_min_dur = st.number_input("Punch Min Duration (frames)", 1, 10, 2)
        punch_gap_tol = st.number_input("Punch Gap Tolerance (frames)", 0, 10, 1)
    with col2:
        kick_min_dur = st.number_input("Kick Min Duration (frames)", 1, 20, 6)
        kick_gap_tol = st.number_input("Kick Gap Tolerance (frames)", 0, 10, 4)
    
    if st.button("🔍 Run Analysis"):
        run_transformer_analysis(
            model_path, csv_path, use_legacy,
            sequence_length, stride,
            punch_threshold, kick_threshold,
            punch_min_dur, punch_gap_tol,
            kick_min_dur, kick_gap_tol
        )
    
    # Save button - works with session state
    st.write("---")
    if st.session_state.get('analysis_complete', False):
        st.info("✅ Analysis results ready to save")
        
        if st.button("💾 Save Results for LLM Analysis", type="primary"):
            try:
                analysis_data = st.session_state.get('analysis_results', {})
                if analysis_data:
                    os.makedirs("runs", exist_ok=True)
                    analysis_path = f"runs/analysis_results_{int(time.time())}.json"
                    with open(analysis_path, 'w') as f:
                        json.dump(analysis_data, f, indent=2)
                    
                    st.success(f"✅ Results saved to `{analysis_path}`")
                    st.info("Go to **LLM Analysis** page in the sidebar to generate AI insights.")
                    
                    # Clear the flag after saving
                    st.session_state['analysis_complete'] = False
                else:
                    st.error("No analysis data found. Please run analysis first.")
            except Exception as e:
                st.error(f"Error saving results: {e}")
    else:
        st.caption("💡 Run analysis first, then the save button will become active.")


# =============================================================================
# OLLAMA LLM MANAGEMENT
# =============================================================================

def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_ollama_server():
    """Start Ollama server in background."""
    try:
        # Check if already running
        if check_ollama_running():
            return True, "Ollama already running"
        
        # Try to start ollama serve
        if sys.platform == "win32":
            # Windows - use subprocess with CREATE_NEW_CONSOLE
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Linux/Mac - use nohup
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Wait for server to start
        for _ in range(10):
            time.sleep(1)
            if check_ollama_running():
                return True, "Ollama server started"
        
        return False, "Ollama server failed to start within 10 seconds"
    
    except FileNotFoundError:
        return False, "Ollama not installed. Install from https://ollama.ai"
    except Exception as e:
        return False, f"Error starting Ollama: {e}"


def get_ollama_models():
    """Get list of locally available Ollama models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("models", []):
                name = model.get("name", "")
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size else 0
                models.append({
                    "name": name,
                    "size_gb": size_gb,
                    "display": f"{name} ({size_gb:.1f} GB)"
                })
            
            # Sort by size (smaller = faster inference)
            models.sort(key=lambda x: x["size_gb"])
            return models
        return []
    except:
        return []


def get_recommended_model(models):
    """Get recommended fast model from available models."""
    # Priority order for fast inference (smaller models first)
    fast_models = [
        "gemma2:2b", "gemma:2b", "phi3:mini", "phi:2.7b", "phi",
        "qwen2:1.5b", "qwen:1.8b", "tinyllama", "tinydolphin",
        "gemma2:9b", "gemma:7b", "llama3.2:3b", "llama3.2:1b",
        "mistral:7b", "llama3:8b", "llama2:7b", "codellama:7b",
        "gemma3:4b", "qwen2.5:7b"
    ]
    
    model_names = [m["name"] for m in models]
    
    for fast_model in fast_models:
        for available in model_names:
            if fast_model in available.lower() or available.lower().startswith(fast_model.split(":")[0]):
                return available
    
    # Return first (smallest) model if no preferred match
    if models:
        return models[0]["name"]
    
    return None


def run_llm_analysis(transformer_punch, transformer_kick, yolo_punch, yolo_kick, 
                     punch_f1=None, kick_f1=None):
    """Run LLM analysis with selected model - called from LLM Analysis page."""
    
    try:
        from ollama import chat
        
        system_prompt = """You are an expert combat-sports analyst. Talk like a human coach giving feedback.
Only use the statistics provided; do not invent or infer any others.
The opponent is always a stationary boxing bag.
Produce exactly five observations:
• Each one sentence, ≤15 words.
• Clearly numbered 1–5.
Do not add any extra text or explanations."""
        
        user_stats = {
            "Transformer_Punches": transformer_punch,
            "Transformer_Kicks": transformer_kick,
            "YOLO_Punches": yolo_punch,
            "YOLO_Kicks": yolo_kick,
        }
        if punch_f1 is not None:
            user_stats["Punch_Detection_F1"] = f"{punch_f1:.3f}"
        if kick_f1 is not None:
            user_stats["Kick_Detection_F1"] = f"{kick_f1:.3f}"
        
        user_prompt = f"Analyze this fighter's bag work session:\n{json.dumps(user_stats, indent=2)}"
        
        return system_prompt, user_prompt, user_stats
        
    except ImportError:
        return None, None, None


def llm_analysis_page():
    """Dedicated page for LLM Analysis with Ollama integration."""
    
    st.title("🧠 LLM Analysis")
    st.markdown("Generate AI-powered insights from your combat sports analysis results.")
    
    # ==========================================================================
    # SECTION 1: Ollama Server Management
    # ==========================================================================
    st.write("---")
    st.write("## 1️⃣ Ollama Server")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        server_status = check_ollama_running()
        if server_status:
            st.success("✅ Ollama server is running")
        else:
            st.warning("⚠️ Ollama server is not running")
    
    with col2:
        if st.button("🔄 Check Status"):
            st.rerun()
    
    with col3:
        if not server_status:
            if st.button("▶️ Start Server"):
                with st.spinner("Starting Ollama server..."):
                    success, message = start_ollama_server()
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    
    if not server_status:
        st.markdown("""
        **Ollama not running?** Try these steps:
        1. Install Ollama from [ollama.ai](https://ollama.ai)
        2. Open a terminal and run: `ollama serve`
        3. Pull a model: `ollama pull gemma2:2b`
        """)
        return
    
    # ==========================================================================
    # SECTION 2: Model Selection
    # ==========================================================================
    st.write("---")
    st.write("## 2️⃣ Select Model")
    
    models = get_ollama_models()
    
    if not models:
        st.warning("No models found. Pull a model first:")
        st.code("ollama pull gemma2:2b", language="bash")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Fast models (recommended):**")
            st.code("ollama pull gemma2:2b\nollama pull phi3:mini\nollama pull llama3.2:1b", language="bash")
        with col2:
            st.markdown("**Larger models (better quality):**")
            st.code("ollama pull llama3:8b\nollama pull mistral:7b", language="bash")
        
        if st.button("🔄 Refresh Model List"):
            st.rerun()
        return
    
    # Model selection with info
    recommended = get_recommended_model(models)
    model_names = [m["name"] for m in models]
    default_idx = model_names.index(recommended) if recommended in model_names else 0
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Select Model (sorted by size - smaller = faster)",
            model_names,
            index=default_idx,
            help="Smaller models have faster inference"
        )
    
    with col2:
        model_info = next((m for m in models if m["name"] == selected_model), None)
        if model_info:
            st.metric("Size", f"{model_info['size_gb']:.1f} GB")
    
    # Show speed estimate
    if model_info:
        size = model_info['size_gb']
        if size < 2:
            st.info("⚡ Very fast inference (~1-3 seconds)")
        elif size < 5:
            st.info("🚀 Fast inference (~3-8 seconds)")
        elif size < 10:
            st.info("🏃 Moderate inference (~8-15 seconds)")
        else:
            st.info("🐢 Slower inference (~15-30+ seconds)")
    
    # ==========================================================================
    # SECTION 3: Load Analysis Data
    # ==========================================================================
    st.write("---")
    st.write("## 3️⃣ Load Analysis Data")
    
    # Find saved analysis results
    analysis_files = []
    if os.path.exists("runs"):
        analysis_files = [f for f in os.listdir("runs") if f.startswith("analysis_results_") and f.endswith(".json")]
        analysis_files.sort(reverse=True)  # Most recent first
    
    # Check for session state data
    has_session_data = st.session_state.get('analysis_results') is not None
    
    data_options = ["Enter manually"]
    if has_session_data:
        data_options.insert(0, "Use current session results")
    if analysis_files:
        data_options.insert(0, "Load from saved file")
    
    data_source = st.radio(
        "Data Source:",
        data_options,
        horizontal=True
    )
    
    analysis_data = None
    
    if data_source == "Use current session results" and has_session_data:
        analysis_data = st.session_state['analysis_results']
        st.success("✅ Using results from current session")
        
        # Display loaded data
        st.write("**Session Data:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Transformer Punches", analysis_data.get('transformer_punch_count', 'N/A'))
            st.metric("Transformer Kicks", analysis_data.get('transformer_kick_count', 'N/A'))
        with col2:
            st.metric("YOLO Punches", analysis_data.get('yolo_punch_count', 'N/A'))
            st.metric("YOLO Kicks", analysis_data.get('yolo_kick_count', 'N/A'))
        
        if analysis_data.get('punch_f1'):
            st.write(f"**Punch F1:** {analysis_data['punch_f1']:.3f} | **Kick F1:** {analysis_data.get('kick_f1', 0):.3f}")
    
    elif data_source == "Load from saved file":
        if not analysis_files:
            st.warning("No saved analysis files found.")
            st.markdown("""
            **To create saved results:**
            1. Go to **Transformer Results** page
            2. Run analysis and click **Save Results**
            """)
        else:
            selected_file = st.selectbox("Select Results File", analysis_files)
            
            if selected_file:
                with open(os.path.join("runs", selected_file)) as f:
                    analysis_data = json.load(f)
                
                # Display loaded data
                st.write("**Loaded Data:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Transformer Punches", analysis_data.get('transformer_punch_count', 'N/A'))
                    st.metric("Transformer Kicks", analysis_data.get('transformer_kick_count', 'N/A'))
                with col2:
                    st.metric("YOLO Punches", analysis_data.get('yolo_punch_count', 'N/A'))
                    st.metric("YOLO Kicks", analysis_data.get('yolo_kick_count', 'N/A'))
                
                if analysis_data.get('punch_f1'):
                    st.write(f"**Punch F1:** {analysis_data['punch_f1']:.3f} | **Kick F1:** {analysis_data.get('kick_f1', 0):.3f}")
                
                st.caption(f"Source: {analysis_data.get('csv_file', 'Unknown')} | Model: {analysis_data.get('model_file', 'Unknown')} | Time: {analysis_data.get('timestamp', 'Unknown')}")
    
    else:  # Manual entry
        st.write("**Enter Analysis Data:**")
        col1, col2 = st.columns(2)
        with col1:
            manual_trans_punch = st.number_input("Transformer Punches", 0, 1000, 0)
            manual_trans_kick = st.number_input("Transformer Kicks", 0, 1000, 0)
        with col2:
            manual_yolo_punch = st.number_input("YOLO Punches", 0, 1000, 0)
            manual_yolo_kick = st.number_input("YOLO Kicks", 0, 1000, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            manual_punch_f1 = st.number_input("Punch F1 (optional)", 0.0, 1.0, 0.0, 0.01)
        with col2:
            manual_kick_f1 = st.number_input("Kick F1 (optional)", 0.0, 1.0, 0.0, 0.01)
        
        analysis_data = {
            'transformer_punch_count': manual_trans_punch,
            'transformer_kick_count': manual_trans_kick,
            'yolo_punch_count': manual_yolo_punch,
            'yolo_kick_count': manual_yolo_kick,
            'punch_f1': manual_punch_f1 if manual_punch_f1 > 0 else None,
            'kick_f1': manual_kick_f1 if manual_kick_f1 > 0 else None
        }
    
    # ==========================================================================
    # SECTION 4: Generate Analysis
    # ==========================================================================
    st.write("---")
    st.write("## 4️⃣ Generate Analysis")
    
    if analysis_data is None:
        st.warning("Please load or enter analysis data first.")
        return
    
    # Custom prompt option
    with st.expander("⚙️ Advanced: Customize Prompt"):
        default_system = """You are an expert combat-sports analyst. Talk like a human coach giving feedback.
Only use the statistics provided; do not invent or infer any others.
The opponent is always a stationary boxing bag.
Produce exactly five observations:
• Each one sentence, ≤15 words.
• Clearly numbered 1–5.
Do not add any extra text or explanations."""
        
        custom_system = st.text_area("System Prompt", default_system, height=150)
        
        analysis_type = st.selectbox(
            "Analysis Style",
            ["Concise (5 bullet points)", "Detailed Report", "Training Recommendations", "Performance Comparison"]
        )
        
        if analysis_type == "Detailed Report":
            custom_system = """You are an expert combat-sports analyst writing a detailed performance report.
Analyze the provided statistics and write a 3-paragraph report covering:
1. Overall performance assessment
2. Strengths identified in the data
3. Areas for improvement
Be specific and use the actual numbers provided."""
        elif analysis_type == "Training Recommendations":
            custom_system = """You are a combat sports coach providing training recommendations.
Based on the statistics provided, give 3-5 specific training recommendations.
Focus on actionable advice the fighter can implement in their next session.
Use the actual numbers to justify your recommendations."""
        elif analysis_type == "Performance Comparison":
            custom_system = """You are a sports analyst comparing model predictions.
Compare the Transformer predictions vs YOLO ground truth.
Discuss what the differences might indicate about:
1. Model accuracy
2. Detection patterns
3. Areas where the models agree/disagree
Keep it technical but accessible."""
    
    # Generate button
    if st.button("🚀 Generate Analysis", type="primary", use_container_width=True):
        try:
            from ollama import chat
            
            # Build the prompt
            user_stats = {
                "Transformer_Punches": analysis_data.get('transformer_punch_count', 0),
                "Transformer_Kicks": analysis_data.get('transformer_kick_count', 0),
                "YOLO_Punches": analysis_data.get('yolo_punch_count', 'N/A'),
                "YOLO_Kicks": analysis_data.get('yolo_kick_count', 'N/A'),
            }
            
            if analysis_data.get('punch_f1'):
                user_stats["Punch_Detection_F1"] = f"{analysis_data['punch_f1']:.3f}"
            if analysis_data.get('kick_f1'):
                user_stats["Kick_Detection_F1"] = f"{analysis_data['kick_f1']:.3f}"
            
            user_prompt = f"Analyze this fighter's bag work session:\n{json.dumps(user_stats, indent=2)}"
            
            st.write("---")
            st.write("### 📝 Analysis Results")
            
            # Time the inference
            t_start = time.perf_counter()
            
            with st.spinner(f"Generating with {selected_model}..."):
                stream = chat(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": custom_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=True,
                )
                
                full_response = ""
                response_placeholder = st.empty()
                
                for chunk in stream:
                    delta = chunk.get("message", {}).get("content", "")
                    full_response += delta
                    response_placeholder.markdown(full_response)
            
            t_end = time.perf_counter()
            
            st.write("---")
            st.caption(f"⏱️ Generated in {t_end - t_start:.1f}s using **{selected_model}**")
            
            # Save option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Copy to Clipboard"):
                    st.code(full_response)
                    st.info("Select and copy the text above")
            
            with col2:
                if st.button("💾 Save Analysis"):
                    save_path = f"runs/llm_analysis_{int(time.time())}.txt"
                    with open(save_path, 'w') as f:
                        f.write(f"Model: {selected_model}\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Stats: {json.dumps(user_stats)}\n")
                        f.write(f"\n{'='*50}\n\n")
                        f.write(full_response)
                    st.success(f"Saved to `{save_path}`")
            
        except ImportError:
            st.error("❌ Ollama Python package not installed")
            st.code("pip install ollama", language="bash")
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            st.info("Try selecting a different model or check if Ollama is running properly.")


def run_transformer_analysis(model_path, csv_path, use_legacy,
                              sequence_length, stride,
                              punch_threshold, kick_threshold,
                              punch_min_dur, punch_gap_tol,
                              kick_min_dur, kick_gap_tol):
    """Run transformer inference and compare with ground truth."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Initialize variables that may not be set in all code paths
    punch_f1 = None
    kick_f1 = None
    
    # Load model
    model_ = load_transformer_model_v2(model_path)
    if model_ is None:
        st.error("Failed to load model")
        return
    
    model_.eval()
    
    # Prepare inputs
    try:
        if use_legacy:
            inputs, labels = prepare_transformer_inputs_legacy(
                csv_path, sequence_length=sequence_length, stride=stride
            )
        else:
            inputs, labels = prepare_transformer_inputs_v2(
                csv_path, sequence_length=sequence_length, stride=stride
            )
    except Exception as e:
        st.error(f"Error preparing inputs: {e}")
        return
    
    if inputs.shape[0] == 0:
        st.warning("No valid sequences found")
        return
    
    st.write(f"**Processing {inputs.shape[0]} sequences...**")
    
    # Run inference with timing
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_.to(device_)
    
    inference_times = []
    
    with torch.no_grad():
        all_logits = []
        
        progress_bar = st.progress(0)
        for i in range(inputs.shape[0]):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            
            logit = model_(inputs[i].unsqueeze(0).to(device_))
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            
            inference_times.append((t_end - t_start) * 1000)
            all_logits.append(logit.cpu())
            
            if i % 10 == 0:
                progress_bar.progress((i + 1) / inputs.shape[0])
        
        progress_bar.progress(1.0)
        logits = torch.cat(all_logits, dim=0)
    
    probs = torch.sigmoid(logits).numpy()
    
    # Aggregate predictions to frame level
    df = pd.read_csv(csv_path)
    total_frames = len(df)
    
    frame_punch_probs = np.zeros(total_frames)
    frame_kick_probs = np.zeros(total_frames)
    frame_counts = np.zeros(total_frames)
    
    for seq_idx in range(probs.shape[0]):
        start_frame = seq_idx * stride
        for t in range(sequence_length):
            frame_idx = start_frame + t
            if frame_idx < total_frames:
                frame_punch_probs[frame_idx] += probs[seq_idx, t, 0]
                frame_kick_probs[frame_idx] += probs[seq_idx, t, 1]
                frame_counts[frame_idx] += 1
    
    # Average overlapping predictions
    frame_counts = np.maximum(frame_counts, 1)
    frame_punch_probs /= frame_counts
    frame_kick_probs /= frame_counts
    
    # Apply thresholds
    punch_hits = frame_punch_probs >= punch_threshold
    kick_hits = frame_kick_probs >= kick_threshold
    
    # Count hits using gap tolerance and min duration (matching YOLO logic)
    def count_events(hit_array, min_dur, gap_tol):
        events = 0
        in_event = False
        event_start = 0
        gap_count = 0
        
        for i, is_hit in enumerate(hit_array):
            if is_hit:
                gap_count = 0
                if not in_event:
                    in_event = True
                    event_start = i
            else:
                if in_event:
                    gap_count += 1
                    if gap_count >= gap_tol:
                        dur = i - event_start - gap_count + 1
                        if dur >= min_dur:
                            events += 1
                        in_event = False
                        gap_count = 0
        
        # Handle event at end
        if in_event:
            dur = len(hit_array) - event_start
            if dur >= min_dur:
                events += 1
        
        return events
    
    transformer_punch_count = count_events(punch_hits, punch_min_dur, punch_gap_tol)
    transformer_kick_count = count_events(kick_hits, kick_min_dur, kick_gap_tol)
    
    # Get ground truth if available
    if not use_legacy:
        gt_punch_hits = df['punch_hit'].values.astype(bool)
        gt_kick_hits = df['kick_hit'].values.astype(bool)
        gt_punch_count = count_events(gt_punch_hits, punch_min_dur, punch_gap_tol)
        gt_kick_count = count_events(gt_kick_hits, kick_min_dur, kick_gap_tol)
        
        # Also get YOLO counts from JSON if available
        base_name = os.path.basename(csv_path).replace('_frame_data.csv', '')
        gt_json_path = f"runs/{base_name}_ground_truth.json"
        if os.path.exists(gt_json_path):
            with open(gt_json_path) as f:
                gt_data = json.load(f)
            yolo_punch_count = gt_data.get('yolo_punch_count', gt_punch_count)
            yolo_kick_count = gt_data.get('yolo_kick_count', gt_kick_count)
        else:
            yolo_punch_count = gt_punch_count
            yolo_kick_count = gt_kick_count
    else:
        gt_punch_hits = None
        gt_kick_hits = None
        gt_punch_count = None
        gt_kick_count = None
        yolo_punch_count = "N/A"
        yolo_kick_count = "N/A"
    
    # Display results
    st.write("---")
    st.write("## 📊 Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Transformer Punches", transformer_punch_count)
        st.metric("Transformer Kicks", transformer_kick_count)
    
    with col2:
        if not use_legacy:
            st.metric("Ground Truth Punches", gt_punch_count)
            st.metric("Ground Truth Kicks", gt_kick_count)
    
    with col3:
        st.metric("YOLO Punches", yolo_punch_count)
        st.metric("YOLO Kicks", yolo_kick_count)
    
    # Accuracy metrics
    if not use_legacy and gt_punch_count is not None:
        st.write("---")
        st.write("### Accuracy Analysis")
        
        # Frame-level accuracy
        punch_accuracy = (punch_hits == gt_punch_hits).mean() * 100
        kick_accuracy = (kick_hits == gt_kick_hits).mean() * 100
        
        # Precision, Recall, F1 for hits
        def compute_metrics(pred, gt):
            tp = ((pred == True) & (gt == True)).sum()
            fp = ((pred == True) & (gt == False)).sum()
            fn = ((pred == False) & (gt == True)).sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            return precision, recall, f1
        
        punch_p, punch_r, punch_f1 = compute_metrics(punch_hits, gt_punch_hits)
        kick_p, kick_r, kick_f1 = compute_metrics(kick_hits, gt_kick_hits)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Punch': [f"{punch_accuracy:.1f}%", f"{punch_p:.3f}", f"{punch_r:.3f}", f"{punch_f1:.3f}"],
            'Kick': [f"{kick_accuracy:.1f}%", f"{kick_p:.3f}", f"{kick_r:.3f}", f"{kick_f1:.3f}"]
        })
        st.table(metrics_df)
        
        # Count accuracy
        punch_count_error = abs(transformer_punch_count - yolo_punch_count) if isinstance(yolo_punch_count, int) else "N/A"
        kick_count_error = abs(transformer_kick_count - yolo_kick_count) if isinstance(yolo_kick_count, int) else "N/A"
        
        st.write(f"**Punch count error:** {punch_count_error}")
        st.write(f"**Kick count error:** {kick_count_error}")
    
    # Inference timing
    st.write("---")
    st.write("### ⏱️ Inference Timing")
    arr = np.array(inference_times)
    st.write(f"- **Mean:** {np.mean(arr):.2f} ms per sequence")
    st.write(f"- **95th Percentile:** {np.percentile(arr, 95):.2f} ms")
    st.write(f"- **Total sequences:** {len(arr)}")
    
    # Timeline visualization
    st.write("---")
    st.write("### 📈 Timeline Visualization")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    frames = np.arange(total_frames)
    
    # Probability plot
    axes[0].plot(frames, frame_punch_probs, label='Punch Prob', color='red', alpha=0.7)
    axes[0].plot(frames, frame_kick_probs, label='Kick Prob', color='blue', alpha=0.7)
    axes[0].axhline(y=punch_threshold, color='red', linestyle='--', alpha=0.5, label=f'Punch Thresh ({punch_threshold})')
    axes[0].axhline(y=kick_threshold, color='blue', linestyle='--', alpha=0.5, label=f'Kick Thresh ({kick_threshold})')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Transformer Hit Probabilities')
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, 1)
    
    # Transformer predictions
    axes[1].fill_between(frames, 0, punch_hits.astype(int), color='red', alpha=0.5, label='Punch Hit')
    axes[1].fill_between(frames, 0, -kick_hits.astype(int), color='blue', alpha=0.5, label='Kick Hit')
    axes[1].set_ylabel('Transformer')
    axes[1].set_title('Transformer Predictions')
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].legend(loc='upper right')
    
    # Ground truth (if available)
    if not use_legacy and gt_punch_hits is not None:
        axes[2].fill_between(frames, 0, gt_punch_hits.astype(int), color='red', alpha=0.5, label='GT Punch')
        axes[2].fill_between(frames, 0, -gt_kick_hits.astype(int), color='blue', alpha=0.5, label='GT Kick')
        axes[2].set_ylabel('Ground Truth')
        axes[2].set_title('Ground Truth (YOLO)')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].legend(loc='upper right')
    else:
        axes[2].text(0.5, 0.5, 'Ground truth not available for legacy CSV format',
                     ha='center', va='center', transform=axes[2].transAxes)
    
    axes[2].set_xlabel('Frame')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Store results in session state for saving
    st.session_state['analysis_results'] = {
        'transformer_punch_count': int(transformer_punch_count),
        'transformer_kick_count': int(transformer_kick_count),
        'yolo_punch_count': int(yolo_punch_count) if isinstance(yolo_punch_count, (int, np.integer)) else str(yolo_punch_count),
        'yolo_kick_count': int(yolo_kick_count) if isinstance(yolo_kick_count, (int, np.integer)) else str(yolo_kick_count),
        'punch_f1': float(punch_f1) if punch_f1 is not None else None,
        'kick_f1': float(kick_f1) if kick_f1 is not None else None,
        'csv_file': os.path.basename(csv_path),
        'model_file': os.path.basename(model_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state['analysis_complete'] = True
    
    st.success("✅ Analysis complete! Results stored. Click the save button below.")


# =============================================================================
# LEGACY FUNCTIONS (for backwards compatibility)
# =============================================================================

def prepare_transformer_inputs_from_csv(csv_file_or_df, sequence_length=None, stride=None, num_classes=None):
    """Legacy wrapper - redirects to appropriate function based on CSV format."""
    DEFAULT_SEQ_LEN = 32
    if sequence_length is None:
        sequence_length = DEFAULT_SEQ_LEN
    if stride is None:
        stride = max(4, sequence_length // 4)
    if num_classes is None:
        num_classes = 6
    
    if isinstance(csv_file_or_df, pd.DataFrame):
        df = csv_file_or_df.copy()
        # Create temp file for processing
        temp_path = "runs/_temp_df.csv"
        df.to_csv(temp_path, index=False)
        csv_path = temp_path
    else:
        csv_path = csv_file_or_df
        df = pd.read_csv(csv_path)
    
    # Check format and use appropriate function
    if 'punch_hit' in df.columns:
        return prepare_transformer_inputs_v2(csv_path, sequence_length, stride)
    else:
        return prepare_transformer_inputs_legacy(csv_path, sequence_length, stride, num_classes)


def load_transformer_model(model_path, d_model=64, nhead=2, num_layers=2,
                           dim_feedforward=128, dropout=0.1, num_classes=None):
    """Legacy wrapper for loading transformer models."""
    return load_transformer_model_v2(model_path)


def compress_state_sequence(actions_sequence):
    if not actions_sequence:
        return []
    compressed = [actions_sequence[0]]
    for i in range(1, len(actions_sequence)):
        if actions_sequence[i] != compressed[-1]:
            compressed.append(actions_sequence[i])
    return compressed


def build_and_run_hmm(action_sequence):
    if not action_sequence:
        return {"message": "No actions seen."}
    states_observed = list(dict.fromkeys(action_sequence))
    transitions_seen = []
    for i in range(len(action_sequence) - 1):
        current_state = action_sequence[i]
        next_state = action_sequence[i + 1]
        transitions_seen.append((current_state, next_state))
    flow_of_states = " -> ".join(action_sequence)
    return {
        "message": "Markov demonstration",
        "states_observed": states_observed,
        "flow_of_states": flow_of_states
    }


def build_segments(hit_set, action, min_len, gap_tol, total_frames):
    segments = []
    in_event = False
    gap_count = 0
    start_f = 0
    for f in range(total_frames):
        is_hit = f in hit_set
        if is_hit:
            gap_count = 0
            if not in_event:
                in_event = True
                start_f = f
        elif in_event:
            gap_count += 1
            if gap_count >= gap_tol:
                dur = f - start_f
                if dur >= min_len:
                    segments.append({
                        'action': action,
                        'start': start_f,
                        'end': f - gap_count,
                        'length': dur
                    })
                in_event = False
                gap_count = 0
    if in_event and (total_frames - start_f) >= min_len:
        segments.append({
            'action': action,
            'start': start_f,
            'end': total_frames - 1,
            'length': total_frames - start_f
        })
    return segments


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.sidebar.title("Combat Sports Analysis")
    app_mode = st.sidebar.selectbox(
        "Choose mode",
        ["Run Model", "Data Collection", "YOLO Training", "Transformer Training", "Transformer Results", "LLM Analysis"]
    )
    
    if app_mode == "Run Model":
        model_execution()
    elif app_mode == "Data Collection":
        data_collection()
    elif app_mode == "YOLO Training":
        yolo_training()
    elif app_mode == "Transformer Training":
        transformer_training_interface()
    elif app_mode == "Transformer Results":
        transformer_results_dashboard()
    elif app_mode == "LLM Analysis":
        llm_analysis_page()


if __name__ == "__main__":
    check_internet_connection()
    main()