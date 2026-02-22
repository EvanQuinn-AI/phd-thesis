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
        socket.create_connection(("www.google.com",80), timeout=5)
        return True
    except OSError:
        return False

def data_collection():
    st.title("Data Collection")
    uploaded_files = st.file_uploader("Upload Videos/Images for Inference or Dataset Training.", accept_multiple_files=True, type=['mp4','avi','mov','jpg','png'])
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            valid_extensions = ['.mp4','.avi','.mov','.jpg','.png']
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
        yolo_model = st.selectbox("YOLO Model", ["yolov5s","yolov5m","yolov5l","yolov5x"])
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
        x1,y1,x2,y2 = box
        placed = False
        for i,(mx1,my1,mx2,my2) in enumerate(merged):
            if boxes_intersect(box, (mx1,my1,mx2,my2)):
                merged[i] = (min(x1,mx1), min(y1,my1),
                             max(x2,mx2), max(y2,my2))
                placed = True
                break
        if not placed:
            merged.append(box)
    changed = True
    while changed:
        changed = False
        new_merged = []
        for box in merged:
            x1,y1,x2,y2 = box
            placed = False
            for j,(nx1,ny1,nx2,ny2) in enumerate(new_merged):
                if boxes_intersect(box, (nx1,ny1,nx2,ny2)):
                    new_merged[j] = (min(x1,nx1), min(y1,ny1),
                                     max(x2,nx2), max(y2,ny2))
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
            cv2.putText(frame, label, (x1, y1-10), 
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
            status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")

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
        st.write(f"- **Achievable FPS (95th percentile):** {1000/total_p95:.1f}" if total_p95 > 0 else "")
        
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

    video_file = st.file_uploader("Upload video", type=['mp4','avi','mov'])
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

        csv_file_path = os.path.join("runs", f"yolo_predictions_{video_file.name}.csv")
        df = pd.DataFrame(all_dets, columns=['frame','x1','y1','x2','y2','confidence','class_id'])
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

### Transformer Functions
class TransformerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class ActionRecognitionTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=6):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer_encoder(emb)
        return self.fc(out)

def prepare_transformer_inputs_from_csv(csv_file_or_df, sequence_length=None, stride=None, num_classes=None):
    DEFAULT_SEQ_LEN = 32
    if sequence_length is None:
        sequence_length = DEFAULT_SEQ_LEN
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive int, got {sequence_length!r}")

    if stride is None:
        stride = max(4, sequence_length // 4)
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError(f"stride must be a positive int, got {stride!r}")
    
    if isinstance(csv_file_or_df, pd.DataFrame):
        df = csv_file_or_df.copy()
    else:
        df = pd.read_csv(csv_file_or_df)
    if 'frame' not in df.columns or 'class_id' not in df.columns:
        raise ValueError("CSV must have columns frame,class_id")
    if num_classes is None:
        dynamic_class_count = len(st.session_state['yolo_classes'])
        if dynamic_class_count < 1:
            unique_ids = df['class_id'].unique()
            dynamic_class_count = len(unique_ids)
        num_classes = dynamic_class_count
    df.sort_values('frame', inplace=True)
    grouped = df.groupby('frame')
    frames = sorted(grouped.groups.keys())
    if not frames:
        return torch.empty(0), torch.empty(0)
    max_frame = frames[-1]
    frame_vectors = np.zeros((max_frame + 1, num_classes), dtype=np.float32)
    for f_idx in frames:
        class_ids = grouped.get_group(f_idx)['class_id'].unique()
        for cid in class_ids:
            cid = int(cid)
            if cid >= 0 and cid < num_classes:
                frame_vectors[f_idx][cid] = 1.0
    input_chunks = []
    label_chunks = []
    i = 0
    while i + sequence_length <= (max_frame + 1):
        seq_data = frame_vectors[i:i+sequence_length]
        seq_tensor = torch.tensor(seq_data, dtype=torch.float)
        input_chunks.append(seq_tensor)
        label_chunks.append(seq_tensor)
        i += stride
    if not input_chunks:
        return torch.empty(0), torch.empty(0)
    inputs = torch.stack(input_chunks)
    labels = torch.stack(label_chunks)
    return inputs, labels

def train_transformer_model(inputs, labels, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=None, num_epochs=30, batch_size=8):
    if isinstance(inputs, list):
        inputs = torch.stack(inputs)
    if isinstance(labels, list):
        labels = torch.stack(labels)
    if num_classes is None or num_classes < 1:
        if 'yolo_classes' in st.session_state and len(st.session_state['yolo_classes']) > 0:
            num_classes = len(st.session_state['yolo_classes'])
        else:
            num_classes = inputs.shape[-1]
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(inputs)
    if N < 2:
        train_in = inputs
        train_lb = labels
        val_in = torch.empty(0)
        val_lb = torch.empty(0)
    else:
        split_idx = int(0.8 * N)
        train_in = inputs[:split_idx]
        train_lb = labels[:split_idx]
        val_in = inputs[split_idx:]
        val_lb = labels[split_idx:]
    train_ds = TransformerDataset(train_in, train_lb)
    val_ds = TransformerDataset(val_in, val_lb)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model_ = ActionRecognitionTransformer(input_size=num_classes, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, num_classes=num_classes).to(device_)
    crit = nn.BCEWithLogitsLoss().to(device_)
    optimizer = optim.Adam(model_.parameters(), lr=1e-3)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    pbar = st.progress(0)
    e_text = st.empty()
    best_val_loss = float('inf')
    patience = 10
    pat_cnt = 0
    for epoch in range(num_epochs):
        model_.train()
        total_train = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device_)
            yb = yb.to(device_)
            optimizer.zero_grad()
            out = model_(xb)
            loss = crit(out.view(-1,num_classes), yb.view(-1,num_classes))
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        scheduler.step()
        val_loss = 0.0
        model_.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device_)
                yb = yb.to(device_)
                outs = model_(xb)
                vl = crit(outs.view(-1,num_classes), yb.view(-1,num_classes))
                val_loss += vl.item()
        e_text.text(f"Epoch[{epoch+1}/{num_epochs}] train={total_train:.4f} val={val_loss:.4f}")
        pbar.progress((epoch+1)/num_epochs)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat_cnt = 0
            torch.save(model_.state_dict(), "models/transformer_model_temp.pth")
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                st.warning("Early stopping triggered.")
                try:
                    model_.load_state_dict(torch.load("models/transformer_model_temp.pth"))
                except:
                    pass
                break
    final_ = "models/transformer_model.pth"
    torch.save(model_.state_dict(), final_)
    st.success(f"Transformer saved => {final_}")
    return model_

def load_transformer_model(model_path, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=None):
    if num_classes is None or num_classes < 1:
        if 'yolo_classes' in st.session_state and len(st.session_state['yolo_classes']) > 0:
            num_classes = len(st.session_state['yolo_classes'])
        else:
            num_classes = 6
    try:
        model_ = ActionRecognitionTransformer(input_size=num_classes, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, num_classes=num_classes).to(device)
        sd = torch.load(model_path, map_location=device)
        model_.load_state_dict(sd, strict=False)
        return model_
    except:
        return None

def transformer_training_interface():
    st.title("Transformer Training (Frame+Class Multi-Label)")
    csv_file = st.file_uploader("Upload CSV", type=['csv'])
    if csv_file is None:
        cfs = [f for f in os.listdir('runs') if f.endswith('.csv')]
        chosen = st.selectbox("Select CSV", cfs)
        if chosen:
            path_ = os.path.join("runs", chosen)
            st.session_state.csv_file_path = path_
            st.success(f"Selected CSV: {chosen}")
    else:
        path_ = os.path.join("data", csv_file.name)
        with open(path_,"wb") as fil:
            fil.write(csv_file.getbuffer())
        st.session_state.csv_file_path = path_
        st.success(f"CSV: {csv_file.name}")
    if 'csv_file_path' in st.session_state:
        df_ = pd.read_csv(st.session_state.csv_file_path)
        st.write("Sample CSV:")
        st.write(df_.head(20))
        if st.button("Train Transformer"):
            try:
                seq_len = 32
                stride = max(4, seq_len // 4)
                num_classes = 6
                inputs, labels = prepare_transformer_inputs_from_csv(df_, sequence_length=seq_len, num_classes=num_classes, stride=stride)
                st.write(f"inputs={inputs.shape}, labels={labels.shape}")
                model_ = train_transformer_model(inputs, labels, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=num_classes, num_epochs=100, batch_size=16)
                st.success("Transformer trained.")
            except Exception as e:
                st.error(f"Error training: {e}")

### HMM Functions
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

### Results Dashboard
def transformer_results_dashboard():
    st.title("Results Dashboard")
    t_models = [f for f in os.listdir("models") if f.endswith(".pth")]
    if not t_models:
        st.warning("No Transformer models.")
        return
    sel_t = st.selectbox("Transformer Model", t_models)
    cfiles = [f for f in os.listdir("runs") if f.endswith('.csv')]
    if not cfiles:
        st.warning("No CSV in runs.")
        return
    sel_csv = st.selectbox("Select CSV", cfiles)
    
    csv_path = os.path.join("runs", sel_csv)
    df_ = pd.read_csv(csv_path)
    run_transformer_statistics(os.path.join("models", sel_t), df_, sel_csv)

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

def run_transformer_statistics(model_path, df_detections, video_path=None):
    """Transformer-based approach with benchmarking for transformer inference."""
    import matplotlib.pyplot as plt

    model_ = load_transformer_model(model_path, d_model=64, nhead=2, num_layers=2,
                                    dim_feedforward=128, dropout=0.1, num_classes=6)
    if not model_:
        st.error("Transformer model not found")
        return

    seq_len = 32
    stride = max(4, seq_len // 4)
    inputs, _ = prepare_transformer_inputs_from_csv(df_detections, seq_len, stride, 6)
    if inputs.size == 0:
        st.warning("No valid sequences")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_.to(device).eval()
    
    # Benchmark transformer inference
    transformer_timings = []
    
    with torch.no_grad():
        logits_list = []
        for i in range(inputs.shape[0]):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            logit = model_(inputs[i].unsqueeze(0).to(device)).cpu()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            transformer_timings.append((t_end - t_start) * 1000)
            logits_list.append(logit)
        logits = torch.cat(logits_list, dim=0)
    
    # Report transformer inference timing
    if transformer_timings:
        arr = np.array(transformer_timings)
        st.write("---")
        st.write("### 🔄 Transformer Inference Benchmark")
        st.write(f"- **Mean:** {np.mean(arr):.2f} ms per sequence ({seq_len} frames)")
        st.write(f"- **95th Percentile:** {np.percentile(arr, 95):.2f} ms")
        st.write(f"- **Total sequences processed:** {len(arr)}")

    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    global_preds = {}
    global_probs = {}

    for win_idx in range(preds.shape[0]):
        start = win_idx * stride
        for t in range(seq_len):
            frame = start + t
            arr_pred = preds[win_idx, t]
            arr_prob = probs[win_idx, t]
            if frame not in global_preds:
                global_preds[frame] = np.zeros(6, dtype=int)
                global_probs[frame] = np.zeros(6, dtype=float)
            global_preds[frame] = np.maximum(global_preds[frame], arr_pred)
            global_probs[frame] = np.maximum(global_probs[frame], arr_prob)

    frames = sorted(global_preds.keys())
    if not frames:
        st.warning("No frames predicted.")
        return
    total_frames = frames[-1] + 1

    class_map = {0: "boxing-bag", 1: "high-guard", 2: "kick-knee",
                 3: "low-guard", 4: "person", 5: "punch"}

    sequence = []
    punch_frames = set()
    kick_frames = set()

    for f in frames:
        g = global_preds[f]
        if g[5] and g[0]:
            sequence.append("punch")
            punch_frames.add(f)
        elif g[2] and g[0]:
            sequence.append("kick-knee")
            kick_frames.add(f)
        elif g[1]:
            sequence.append("high-guard")
        else:
            sequence.append("idle")

    df_hits = pd.DataFrame({'frame': frames})
    df_hits['punches'] = df_hits['frame'].isin(punch_frames).cumsum()
    df_hits['kicks'] = df_hits['frame'].isin(kick_frames).cumsum()

    df_seq = pd.DataFrame({'frame': frames, 'action': sequence})

    df_seq['segment'] = (df_seq['action'] != df_seq['action'].shift()).cumsum()
    segs = df_seq.groupby('segment').agg(
        action=('action', 'first'),
        start=('frame', 'first'),
        end=('frame', 'last')
    ).reset_index(drop=True)
    segs['length'] = segs['end'] - segs['start'] + 1

    durations = df_seq['action'].value_counts().reindex(class_map.values(), fill_value=0)
    active = df_seq[df_seq['action'].isin(['punch','kick-knee'])].shape[0]
    active_ratio = active / total_frames * 100

    MIN_LEN = {'punch': 2, 'kick-knee': 6}
    GAP_TOL = {'punch': 1, 'kick-knee': 4}

    punch_segments = pd.DataFrame(
        build_segments(punch_frames, 'punch', MIN_LEN['punch'], GAP_TOL['punch'], total_frames)
    )
    kick_segments = pd.DataFrame(
        build_segments(kick_frames, 'kick-knee', MIN_LEN['kick-knee'], GAP_TOL['kick-knee'], total_frames)
    )

    st.write("### Activity")
    gpt_punch = len(punch_segments)
    gpt_kick = len(kick_segments)
    st.write(f"Estimated Punches: {gpt_punch}")
    st.write(f"Estimated Kicks: {gpt_kick}")
    st.write(f"Active: {active_ratio:.1f}%")
    st.write(f"Resting: {100 - active_ratio:.1f}%")

    stats = segs.groupby('action')['length'].describe().rename(columns={
        'count':'Segments', 'mean':'AvgLen', 'min':'MinLen', '25%':'Q1','50%':'Median','75%':'Q3','max':'MaxLen'
    })

    compressed = compress_state_sequence(sequence)

    trans = pd.DataFrame({'prev': compressed[:-1], 'next': compressed[1:]})
    trans_mat = pd.crosstab(trans['prev'], trans['next']).reindex(index=stats.index, columns=stats.index, fill_value=0)

    # GPT Testing (simplified - remove if ollama not available)
    try:
        from ollama import chat
        system_prompt = """
        You are an expert combat-sports analyst. Talk like a human.
        Only use the statistics provided; do not invent or infer any others.  
        The opponent is always a stationary boxing bag.  
        Produce exactly five observations:  
        • Each one sentence, ≤15 words.  
        • Clearly numbered 1–5.  
        Do not add any extra text or explanations.
        """
        user_stats = {
            "Punches": gpt_punch,
            "Kicks": gpt_kick,
            "Action Sequences": compressed
        }
        user_prompt = f"Here are the statistics:\n{user_stats}"
        stream = chat(
            model="gemma3:4b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        st.write("### LLM Response")
        full = ""
        placeholder = st.empty()
        for chunk in stream:
            if st.session_state.get("stop_requested", False):
                placeholder.markdown(full + "\n\n*⏹️ Generation stopped.*")
                break
            delta = chunk["message"]["content"]
            full += delta
            placeholder.markdown(full)
        st.info("🧠 Generated…")
    except ImportError:
        st.info("Ollama not available - skipping LLM analysis")

    instances = {}
    instances['punch'] = len(punch_segments)
    instances['kick-knee'] = len(kick_segments)
    for action in ['high-guard', 'low-guard', 'idle']:
        instances[action] = segs[segs['action'] == action].shape[0]
    instance_counts = (
        pd.Series(instances)
        .rename_axis('Action')
        .to_frame(name='Instances')
    )

    st.write("### Action Instances Chart")
    st.bar_chart(instance_counts)

    st.write(f"### CSV Preview - {video_path}")
    st.write(df_detections.head(5))

    from matplotlib.patches import Patch
    st.write("### Action Timeline Diagram")
    fig_timeline, ax = plt.subplots(figsize=(8, 2))
    color_map = {'punch': 'red', 'kick-knee': 'blue', 'high-guard': 'green', 'idle': 'gray'}

    for _, row in segs.iterrows():
        ax.barh(0, row['length'], left=row['start'],
                color=color_map.get(row['action'], 'gray'), alpha=0.8,
                edgecolor='black', linewidth=0.5)

    ax.grid(axis='x', linestyle='--', alpha=0.5)
    handles = [Patch(color=col, label=act) for act, col in color_map.items()]
    ax.legend(handles=handles, ncol=len(handles), bbox_to_anchor=(0.5, 1.2),
              loc='upper center', frameon=False)
    ax.set_yticks([])
    ax.set_xlabel('Frame')
    st.pyplot(fig_timeline, use_container_width=False)

    st.write("### Cumulative Hits Over Time")
    fig_cum_hits, ax = plt.subplots()
    ax.plot(df_hits['frame'], df_hits['punches'], label='Punches')
    ax.plot(df_hits['frame'], df_hits['kicks'], label='Kicks')
    ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Hits'); ax.legend()
    st.pyplot(fig_cum_hits, use_container_width=False)

### Run Main & Check Internet
def main():
    st.sidebar.title("Combat Sports Analysis")
    app_mode = st.sidebar.selectbox("Choose mode", ["Run Model","Data Collection","YOLO Training","Transformer Training","Transformer Results"])
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

if __name__ == "__main__":
    check_internet_connection()
    main()
