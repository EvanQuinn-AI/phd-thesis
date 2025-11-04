# Important Imports
import os
import shutil
import sys
import subprocess
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
import plotly.express as px
import streamlit as st
import cv2
from torch.utils.data import Dataset, DataLoader
from collections import deque
import threading
from queue import Queue
from PIL import Image
import io
from ultralytics import YOLO

# Error Handling
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
torch.classes.__path__ = []   # neutralize the broken proxy
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*figure.*")
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent figure warnings

# Check if Device is CUDA ready
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("CUDA: ", torch.version.cuda, torch.cuda.is_available())

# Avoid Logging Issues
logging.getLogger("torch").setLevel(logging.ERROR)
st.set_page_config(page_title="Combat Sports Prototype", layout="wide", page_icon="🥊")
pathlib.PosixPath = pathlib.WindowsPath

# Apply modern professional styling
def apply_custom_css():
    """Apply modern, professional CSS styling to the entire app"""
    st.markdown("""
    <style>
    /* Modern Color Palette */
    :root {
        --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --primary-solid: #667eea;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --info: #3B82F6;
        --dark-bg: #1e1e2e;
        --card-bg: #2a2a3e;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
    }
    
    /* Main App Styling */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: var(--text-primary);
    }
    
    /* Card Styling */
    .card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4), 0 2px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Status Badge Styling */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 4px;
    }
    
    .badge-success {
        background: var(--success);
        color: white;
    }
    
    .badge-warning {
        background: var(--warning);
        color: white;
    }
    
    .badge-error {
        background: var(--error);
        color: white;
    }
    
    .badge-info {
        background: var(--info);
        color: white;
    }
    
    /* Modern Button Styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Typography Improvements */
    h1 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
    }
    
    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(30, 30, 46, 0.8);
    }
    
    /* Metrics/Statistics Styling */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Video Container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Statistics Display */
    .stats-display {
        background: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    /* AI Analysis Card */
    .ai-analysis-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 10px 0;
    }
    
    /* Info Messages Styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Input Fields */
    .stSelectbox > div > div {
        background: var(--card-bg);
        border-radius: 8px;
    }
    
    /* File Uploader */
    .stFileUploader {
        border-radius: 8px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Blinking cursor animation for streaming text */
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Logo pulse animation */
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.9; }
    }
    
    @keyframes glow {
        0%, 100% { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5)); }
        50% { filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.8)); }
    }
    
    /* Logo card styling */
    .logo-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s, box-shadow 0.2s;
        min-height: 80px;
    }
    
    .logo-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4), 0 2px 6px rgba(0, 0, 0, 0.3);
    }
    
    .logo-card .logo {
        font-size: 3em;
        animation: pulse 2s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.6));
    }
    
    /* Hide Streamlit header buttons (Deploy and empty header button) */
    button[data-testid="stBaseButton-header"],
    button[data-testid="stBaseButton-headerNoPadding"] {
        display: none !important;
    }
    
    /* Hide Sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hide specific Streamlit emotion cache element */
    .st-emotion-cache-aak2an.e1ycw9pz3 {
        display: none !important;
    }
    
    /* Expand main content to full width */
    .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Code tag styling */
    code {
        background: rgba(255, 255, 255, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        color: #10B981;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize custom CSS on app load
apply_custom_css()

# Reusable UI Components
def status_card(title, value, status_type="info", icon=""):
    """Create a styled status card component"""
    badge_class = f"badge-{status_type}"
    return st.markdown(f"""
    <div class="card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="color: var(--text-secondary); font-size: 0.9em; margin-bottom: 5px;">{title}</div>
                <div style="font-size: 1.5em; font-weight: 700; color: var(--text-primary);">
                    {icon} {value}
                </div>
            </div>
            <span class="status-badge {badge_class}">{status_type.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def info_card(title, content, icon=""):
    """Create a styled info card"""
    return st.markdown(f"""
    <div class="card">
        <h4 style="margin-top: 0; color: var(--text-primary);">{icon} {title}</h4>
        <div style="color: var(--text-secondary);">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_status_badge(text, status_type="info"):
    """Create a status badge"""
    badge_class = f"badge-{status_type}"
    return f'<span class="status-badge {badge_class}">{text}</span>'

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
# Real-time analysis session state
if "realtime_active" not in st.session_state:
    st.session_state["realtime_active"] = False
if "frame_buffer" not in st.session_state:
    st.session_state["frame_buffer"] = deque(maxlen=32)
if "action_sequence" not in st.session_state:
    st.session_state["action_sequence"] = []
if "last_llm_update" not in st.session_state:
    st.session_state["last_llm_update"] = 0
if "current_statistics" not in st.session_state:
    st.session_state["current_statistics"] = {"punch": 0, "kick-knee": 0, "total_hits": 0, "active_ratio": 0}
if "selected_yolo_model_path" not in st.session_state:
    st.session_state["selected_yolo_model_path"] = None

### Helper Functions
# Make sure internet is available to get YOLO model
def check_internet_connection():
    try:
        socket.create_connection(("www.google.com",80), timeout=5)
        return True
    except OSError:
        return False

# Allows users to upload new data or links to labelling tools
def data_collection():
    # Upload files to the project
    st.title("📁 Data Collection")
    info_card("📤 Upload Media Files", "Upload videos or images for inference or dataset training")
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
    # Links to labeling tools
    info_card("🔗 Labeling Tools", """
    - [Roboflow](https://roboflow.com/) - Cloud-based annotation platform
    - [LabelImg](https://github.com/tzutalin/labelImg) - Desktop annotation tool
    - [CVAT](https://github.com/openvinotoolkit/cvat) - Computer Vision Annotation Tool
    """)

# Loads Classes from YAML
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

# Helper Function to Update YAML Files Path
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

# Loads YOLOv11 model with GPU optimization
@st.cache_resource
def load_yolo_model(weights='models/best.pt'):
    """
    Load and optimize YOLOv11 model for inference.
    Model is cached and optimized for GPU if available.
    YOLOv11 handles device placement and optimization automatically.
    """
    try:
        # Load YOLOv11 model using ultralytics
        model = YOLO(weights)
        
        # YOLOv11 automatically handles:
        # - Device placement (GPU/CPU)
        # - Model evaluation mode
        # - Optimization settings
        
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Automatically selects the best YOLO model
def select_best_yolo_model():
    """
    Selects the best YOLO model:
    1. Prefers 'best.pt' if exists
    2. Falls back to newest .pt file by modification time
    Returns the path to the selected model or None
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if not model_files:
        return None
    
    # Prefer best.pt if it exists
    if 'best.pt' in model_files:
        model_path = os.path.join(models_dir, 'best.pt')
        st.session_state["selected_yolo_model_path"] = model_path
        return model_path
    
    # Fallback to newest file by modification time
    model_paths = [os.path.join(models_dir, f) for f in model_files]
    newest_model = max(model_paths, key=lambda p: os.path.getmtime(p))
    st.session_state["selected_yolo_model_path"] = newest_model
    return newest_model

# Function to train new YOLO models
def yolo_training():
    st.title("🎯 YOLO Model Training")
    info_card("📂 Dataset Configuration", "Upload or select your dataset configuration file")
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
        yolo_model = st.selectbox("YOLO Model", ["yolo11n","yolo11s","yolo11m","yolo11l","yolo11x"])
        if st.button("🚀 Start YOLO Training", type="primary"):
            if not os.path.exists(dataset_path):
                st.error(f"No dataset: {dataset_path}")
                return
            if not os.path.exists(data_yaml_path):
                st.error(f"No YAML: {data_yaml_path}")
                return
            
            status_text = st.empty()
            
            try:
                with st.spinner('Loading YOLOv11 model...'):
                    # Load pretrained YOLOv11 model
                    model = YOLO(f'{yolo_model}.pt')
                    
                status_text.info(f"Training {yolo_model} on dataset for {epochs} epochs...")
                
                # Train the model using Python API
                # Note: YOLOv11 training progress is displayed in the terminal/console
                # Streamlit spinner will show while training is in progress
                with st.spinner('Training YOLOv11 model... This may take a while. Check console for detailed progress.'):
                    train_results = model.train(
                        data=data_yaml_path,
                        epochs=epochs,
                        imgsz=img_size,
                        batch=batch_size,
                        device="0" if torch.cuda.is_available() else "cpu"
                    )
                
                # Evaluate the model
                status_text.info("Evaluating model performance...")
                metrics = model.val()
                
                st.success("✅ YOLOv11 training completed successfully!")
                
                # Get model save path with error handling
                try:
                    if hasattr(model, 'trainer') and hasattr(model.trainer, 'best'):
                        best_path = model.trainer.best
                        st.info(f"Model saved to: {best_path}")
                    else:
                        st.info("Model training completed. Check 'models' directory for saved weights.")
                except Exception as e:
                    st.info("Model training completed. Check 'models' directory for saved weights.")
                
                # Display key metrics with proper validation
                try:
                    if hasattr(metrics, 'box') and metrics.box:
                        if hasattr(metrics.box, 'map50'):
                            st.write(f"**mAP50:** {metrics.box.map50:.4f}")
                        if hasattr(metrics.box, 'map'):
                            st.write(f"**mAP50-95:** {metrics.box.map:.4f}")
                except Exception as e:
                    st.warning(f"Could not display metrics: {str(e)}")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    else:
        st.warning("No valid dataset or data.yaml")

# Checks overlap amount for boxes
def check_overlap(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    return not (x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1)

# Checks if boxes intersect
def boxes_intersect(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    if xA2 < xB1 or xB2 < xA1 or yA2 < yB1 or yB2 < yA1:
        return False
    return True

# Calculates IOU for two boxes
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

# Runs YOLOv11 inference and returns Detections (optimized for GPU)
def yolo_inference(frame, model, max_size=640):
    """
    Optimized YOLOv11 inference with GPU support.
    Resizes frame for faster processing while maintaining aspect ratio.
    Returns detections in format [x1, y1, x2, y2, conf, cls] for compatibility.
    """
    # Resize frame for faster inference (maintains aspect ratio)
    h, w = frame.shape[:2]
    original_h, original_w = h, w
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = frame.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Run YOLOv11 inference
        results = model(rgb_frame, conf=0.1, iou=0.2, verbose=False)
        
        # Extract results from YOLOv11 format
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        
        # Get boxes, confidences, and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] format: [x1, y1, x2, y2]
        confidences = results[0].boxes.conf.cpu().numpy()  # [N] format
        class_ids = results[0].boxes.cls.cpu().numpy()  # [N] format
        
        # Combine into [x1, y1, x2, y2, conf, cls] format
        detections = np.column_stack([boxes, confidences, class_ids])
        
        # Scale detections back to original frame size if we resized
        if max(original_h, original_w) > max_size:
            scale = max(original_h, original_w) / max_size
            detections[:, :4] = detections[:, :4] * scale
        
        # Validate detection array shape
        if len(detections.shape) != 2 or detections.shape[1] != 6:
            print(f"[YOLOv11 Inference Warning] Unexpected detection shape: {detections.shape}, expected [N, 6]")
            return []
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"[YOLOv11 Inference Error] {error_type}: {error_msg}")
        return []
    return detections

# Process Video and Draw Bounding Boxes
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

def process_video(video_path, model):
    class_names = st.session_state.get('yolo_classes', [])
    if not class_names:
        st.error("No class names.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening {video_path}")
        return []

    os.makedirs("runs", exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base   = os.path.splitext(os.path.basename(video_path))[0]
    out    = cv2.VideoWriter(f"runs/{base}.mp4", fourcc, fps, (w,h))

    # state
    prev_frame_time = time.time()
    frame_count    = 0
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counters       = {'punch': 0, 'kick-knee': 0}
    in_event       = {'punch': False, 'kick-knee': False}
    event_start    = {'punch': 0,     'kick-knee': 0}
    min_event_dur  = {'punch': 2,     'kick-knee': 6}
    gap_counter    = {'punch': 0,     'kick-knee': 0}
    gap_tolerance  = {'punch': 1,     'kick-knee': 4}

    progress_bar   = st.progress(0)
    all_detections = []

    def check_overlap_(action_boxes, bag_boxes, frame_):
        for a in action_boxes:
            for b in bag_boxes:
                if boxes_intersect(a, b):
                    ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                    cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                    cv2.line(frame_, ca, cb, (0,255,255), 2)
                    return True
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- compute live FPS ---------------------------------
        new_frame_time = time.time()
        fps = 1.0 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # -------------------------------------------------------

        # 1) YOLO + filter
        dets     = yolo_inference(frame, model)
        CONF_THR = 0.4
        filtered = [d for d in dets if len(d) >= 6 and d[4] >= CONF_THR]

        # 2) build & merge boxes
        raw_bag   = [tuple(map(int,d[:4])) for d in filtered if int(d[5])==0]
        raw_punch = [tuple(map(int,d[:4])) for d in filtered if int(d[5])==5]
        raw_kick  = [tuple(map(int,d[:4])) for d in filtered if int(d[5])==2]

        bag_boxes   = merge_overlapping_boxes(raw_bag)
        punch_boxes = merge_overlapping_boxes(raw_punch)
        kick_boxes  = merge_overlapping_boxes(raw_kick)

        # draw raw detections *with* confidences
        for x1f,y1f,x2f,y2f,conf,cls in filtered:
            cls = int(cls)
            color = {0:(0,255,0),5:(255,0,255),2:(0,255,255)}.get(cls, (0,0,255))
            label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            x1,y1,x2,y2 = map(int,(x1f,y1f,x2f,y2f))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # then draw merged boxes (no text)
        for cls_id, boxes in [(0, bag_boxes), (5, punch_boxes), (2, kick_boxes)]:
            color = {0:(0,255,0),5:(255,0,255),2:(0,255,255)}[cls_id]
            for x1,y1,x2,y2 in boxes:
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        # 4) draw other classes
        for x1f,y1f,x2f,y2f,conf,cls in filtered:
            cls = int(cls)
            if cls not in (0,2,5):
                x1,y1,x2,y2 = map(int,(x1f,y1f,x2f,y2f))
                color = {4:(255,0,0),1:(0,128,255),3:(255,165,0)}.get(cls,(0,0,255))
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                cv2.putText(frame,f"{label} {conf:.2f}",
                            (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        # 5) check overlap
        ov_punch   = check_overlap_(punch_boxes, bag_boxes,   frame)
        ov_kickkne = check_overlap_(kick_boxes,  bag_boxes,   frame)

        # 6) event + gap tolerance logic
        for action, is_over in [('punch',ov_punch),('kick-knee',ov_kickkne)]:
            if is_over:
                # reset gap counter; start event if needed
                gap_counter[action] = 0
                if not in_event[action]:
                    in_event[action]    = True
                    event_start[action] = frame_count

            else:
                if in_event[action]:
                    gap_counter[action] += 1
                    # only close event after N missed frames
                    if gap_counter[action] >= gap_tolerance[action]:
                        dur = frame_count - event_start[action]
                        if dur >= min_event_dur[action]:
                            counters[action] += 1
                        in_event[action] = False
                        gap_counter[action] = 0

        # 7) annotate & write
        total = counters['punch'] + counters['kick-knee']
        cv2.putText(frame,f"Hits: {total}",   (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame,f"Punch: {counters['punch']}",(50,90),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
        cv2.putText(frame,f"Kick-Knee: {counters['kick-knee']}",(50,130),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        # — draw FPS top-right —
        text       = f"FPS: {fps:.1f}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 2
        margin     = 10

        # measure text size so we can right-align
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = frame.shape[1] - text_w - margin            # right-align
        y = margin + text_h                              # a little below the top edge

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            lineType=cv2.LINE_AA
        )

        out.write(frame)
        if filtered:
            all_detections.append(dets)

        frame_count += 1
        progress_bar.progress(min(frame_count/total_frames,1.0))

    # close any still-open events at end
    for action in ('punch','kick-knee'):
        if in_event[action]:
            dur = frame_count - event_start[action]
            if dur >= min_event_dur[action]:
                counters[action] += 1

    cap.release()
    out.release()
    progress_bar.empty()
    st.success(f"Done.\nPunch: {counters['punch']}\nKick-Knee: {counters['kick-knee']}")

    return all_detections

# Initialize real-time processing state
if "realtime_counters" not in st.session_state:
    st.session_state["realtime_counters"] = {'punch': 0, 'kick-knee': 0}
if "realtime_in_event" not in st.session_state:
    st.session_state["realtime_in_event"] = {'punch': False, 'kick-knee': False}
if "realtime_event_start" not in st.session_state:
    st.session_state["realtime_event_start"] = {'punch': 0, 'kick-knee': 0}
if "realtime_frame_count" not in st.session_state:
    st.session_state["realtime_frame_count"] = 0
if "realtime_active_frames" not in st.session_state:
    st.session_state["realtime_active_frames"] = 0
if "realtime_gap_counter" not in st.session_state:
    st.session_state["realtime_gap_counter"] = {'punch': 0, 'kick-knee': 0}
if "realtime_prev_time" not in st.session_state:
    st.session_state["realtime_prev_time"] = time.time()
if "realtime_transformer_counter" not in st.session_state:
    st.session_state["realtime_transformer_counter"] = 0
if "realtime_video_source" not in st.session_state:
    st.session_state["realtime_video_source"] = None
if "camera_type" not in st.session_state:
    st.session_state["camera_type"] = None
if "llm_result_queue" not in st.session_state:
    st.session_state["llm_result_queue"] = Queue()
if "llm_analysis_thread" not in st.session_state:
    st.session_state["llm_analysis_thread"] = None
if "llm_last_result" not in st.session_state:
    st.session_state["llm_last_result"] = "AI Analysis will appear here..."
if "session_final_stats" not in st.session_state:
    st.session_state["session_final_stats"] = None
if "analysis_generating" not in st.session_state:
    st.session_state["analysis_generating"] = False
if "analysis_stream_chunks" not in st.session_state:
    st.session_state["analysis_stream_chunks"] = []
if "browser_frame_batch_count" not in st.session_state:
    st.session_state["browser_frame_batch_count"] = 0
if "analysis_chunks_queue" not in st.session_state:
    st.session_state["analysis_chunks_queue"] = Queue()
    print(f"[Queue Init] Created new analysis_chunks_queue instance: id={id(st.session_state['analysis_chunks_queue'])}")
else:
    queue_instance = st.session_state["analysis_chunks_queue"]
    print(f"[Queue Init] Using existing analysis_chunks_queue instance: id={id(queue_instance)}")
if "rerun_lock" not in st.session_state:
    st.session_state["rerun_lock"] = False

# Utility function to convert camera_input image to OpenCV format
def image_to_cv2(image_input):
    """Convert Streamlit camera_input image (PIL/numpy) to OpenCV BGR format"""
    if image_input is None:
        return None
    
    # Handle PIL Image
    if isinstance(image_input, Image.Image):
        # Convert PIL to numpy array
        img_array = np.array(image_input)
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        return img_bgr
    
    # Handle numpy array
    if isinstance(image_input, np.ndarray):
        # If already in BGR format, return as is
        if len(image_input.shape) == 3:
            # Check if it's RGB (common from camera_input)
            return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if image_input.shape[2] == 3 else image_input
        return image_input
    
    # Handle bytes
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    return None

# Browser camera frame processing function
def process_browser_camera_frame(frame_cv2, yolo_model, transformer_model, video_placeholder, stats_placeholder, graph_placeholder, llm_placeholder):
    """
    Process a single frame from browser camera (st.camera_input).
    frame_cv2: OpenCV BGR image (numpy array)
    Returns: True if successful
    """
    if frame_cv2 is None:
        return False
    
    class_names = st.session_state.get('yolo_classes', [])
    if not class_names:
        class_names = ["boxing-bag", "high-guard", "kick-knee", "low-guard", "person", "punch"]
    
    num_classes = len(class_names)
    
    # Constants
    min_event_dur = {'punch': 2, 'kick-knee': 6}
    gap_tolerance = {'punch': 1, 'kick-knee': 4}
    transformer_stride = 8
    
    # Get frame count
    frame_count = st.session_state["realtime_frame_count"]
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1.0 / (new_frame_time - st.session_state["realtime_prev_time"] + 1e-6)
    st.session_state["realtime_prev_time"] = new_frame_time
    
    # YOLO inference
    dets = yolo_inference(frame_cv2, yolo_model, max_size=640)
    CONF_THR = 0.4
    filtered = [d for d in dets if d[4] >= CONF_THR]
    
    # Build and merge boxes
    raw_bag = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 0]
    raw_punch = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 5]
    raw_kick = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 2]
    
    bag_boxes = merge_overlapping_boxes(raw_bag)
    punch_boxes = merge_overlapping_boxes(raw_punch)
    kick_boxes = merge_overlapping_boxes(raw_kick)
    
    # Check overlaps
    ov_punch = False
    for a in punch_boxes:
        for b in bag_boxes:
            if boxes_intersect(a, b):
                ov_punch = True
                break
        if ov_punch:
            break
    
    ov_kickkne = False
    for a in kick_boxes:
        for b in bag_boxes:
            if boxes_intersect(a, b):
                ov_kickkne = True
                break
        if ov_kickkne:
            break
    
    # Track action frames (frames with punch/kick detections)
    if ov_punch or ov_kickkne:
        st.session_state["realtime_active_frames"] = st.session_state.get("realtime_active_frames", 0) + 1
    
    # Event counting logic
    counters = st.session_state["realtime_counters"]
    in_event = st.session_state["realtime_in_event"]
    event_start = st.session_state["realtime_event_start"]
    gap_counter = st.session_state["realtime_gap_counter"]
    
    for action, is_over in [('punch', ov_punch), ('kick-knee', ov_kickkne)]:
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
    
    # Prepare frame vector for transformer
    frame_vector = np.zeros(num_classes, dtype=np.float32)
    for x1f, y1f, x2f, y2f, conf, cls in filtered:
        cls_id = int(cls)
        if 0 <= cls_id < num_classes:
            frame_vector[cls_id] = 1.0
    
    # Add to frame buffer and process transformer chunk periodically
    st.session_state["realtime_transformer_counter"] += 1
    if st.session_state["realtime_transformer_counter"] >= transformer_stride:
        st.session_state["frame_buffer"].append(frame_vector)
        
        if transformer_model and len(st.session_state["frame_buffer"]) >= 32:
            buffer_count = len(st.session_state["frame_buffer"])
            if buffer_count % 32 == 0:
                actions = process_transformer_chunk(st.session_state["frame_buffer"], transformer_model, num_classes)
                if actions:
                    st.session_state["action_sequence"].extend(actions)
        
        st.session_state["realtime_transformer_counter"] = 0
    
    # Update statistics
    total_hits = counters['punch'] + counters['kick-knee']
    # Calculate active ratio from actual action frames vs total frames
    action_frames = st.session_state.get("realtime_active_frames", 0)
    total_frames = st.session_state["realtime_frame_count"]
    active_ratio = (action_frames / total_frames) * 100 if total_frames > 0 else 0
    
    st.session_state["current_statistics"] = {
        "punch": counters['punch'],
        "kick-knee": counters['kick-knee'],
        "total_hits": total_hits,
        "active_ratio": active_ratio
    }
    
    # Create display frame with annotations
    display_frame = frame_cv2.copy()
    
    # Draw bounding boxes
    for x1f, y1f, x2f, y2f, conf, cls in filtered:
        cls = int(cls)
        color = {0:(0,255,0), 5:(255,0,255), 2:(0,255,255)}.get(cls, (0,0,255))
        label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw overlap lines
    if ov_punch:
        for a in punch_boxes:
            for b in bag_boxes:
                if boxes_intersect(a, b):
                    ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                    cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                    cv2.line(display_frame, ca, cb, (0,255,255), 2)
    if ov_kickkne:
        for a in kick_boxes:
            for b in bag_boxes:
                if boxes_intersect(a, b):
                    ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                    cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                    cv2.line(display_frame, ca, cb, (0,255,255), 2)
    
    # Annotate statistics on frame
    cv2.putText(display_frame, f"Hits: {total_hits}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Punch: {counters['punch']}", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(display_frame, f"Kick-Knee: {counters['kick-knee']}", (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Draw FPS
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = display_frame.shape[1] - text_w - margin
    y = margin + text_h
    cv2.putText(display_frame, text, (x, y), font, font_scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    
    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    # Only show active ratio if video is not active (stopped)
    active_ratio_display = ""
    if not st.session_state.get("realtime_active", True):
        active_ratio_display = f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: var(--text-secondary);">⚡ Active Ratio:</span>
                <span style="font-weight: 600; color: var(--text-primary);">{active_ratio:.1f}%</span>
            </div>
    """
    
    # Update statistics display
    stats_text = f"""
    <div class="stats-display">
        <h4 style="margin-top: 0; color: var(--text-primary); margin-bottom: 15px;">📊 Real-Time Statistics</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
            <div style="background: rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid var(--success);">
                <div style="color: var(--text-secondary); font-size: 0.85em;">Total Hits</div>
                <div style="font-size: 2em; font-weight: 700; color: var(--success);">{total_hits}</div>
            </div>
            <div style="background: rgba(59, 130, 246, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid var(--info);">
                <div style="color: var(--text-secondary); font-size: 0.85em;">FPS</div>
                <div style="font-size: 2em; font-weight: 700; color: var(--info);">{fps:.1f}</div>
            </div>
        </div>
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: var(--text-secondary);">🥊 Punches:</span>
                <span style="font-weight: 600; color: var(--text-primary);">{counters['punch']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: var(--text-secondary);">🦵 Kicks:</span>
                <span style="font-weight: 600; color: var(--text-primary);">{counters['kick-knee']}</span>
            </div>
            {active_ratio_display}
        </div>
    </div>
    """
    stats_placeholder.markdown(stats_text, unsafe_allow_html=True)
    
    # Don't update graph during video playback - graph will be generated only at end
    # Graph placeholder should already show initial message from initialization
    # This prevents flickering during video processing
    
    # Check for LLM results from background thread
    try:
        if not st.session_state["llm_result_queue"].empty():
            llm_result = st.session_state["llm_result_queue"].get_nowait()
            st.session_state["llm_last_result"] = llm_result
    except:
        pass
    
    # Don't update llm_placeholder here - let main function handle it based on video state
    # This prevents conflicts and ensures AI analysis only shows after video stops
    
    st.session_state["realtime_frame_count"] += 1
    return True

# Real-time stream processing function (processes one frame per call)
def process_realtime_frame(yolo_model, transformer_model, video_placeholder, stats_placeholder, graph_placeholder, llm_placeholder):
    """
    Process one frame from the video stream. Optimized for performance.
    - Skips frames for display (shows every Nth frame)
    - Processes smaller images for faster YOLO inference
    - Updates LLM analysis asynchronously
    """
    video_source = st.session_state.get("realtime_video_source")
    if not video_source:
        return False
    
    class_names = st.session_state.get('yolo_classes', [])
    if not class_names:
        class_names = ["boxing-bag", "high-guard", "kick-knee", "low-guard", "person", "punch"]
    
    num_classes = len(class_names)
    
    # Constants
    min_event_dur = {'punch': 2, 'kick-knee': 6}
    gap_tolerance = {'punch': 1, 'kick-knee': 4}
    transformer_stride = 8
    
    def check_overlap_(action_boxes, bag_boxes, frame_):
        for a in action_boxes:
            for b in bag_boxes:
                if boxes_intersect(a, b):
                    ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                    cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                    cv2.line(frame_, ca, cb, (0,255,255), 2)
                    return True
        return False
    
    # Get frame count
    frame_count = st.session_state["realtime_frame_count"]
    
    # Read next frame
    ret, frame = video_source.read()
    if not ret:
        # Just signal end of video - let main loop handle cleanup and stats preservation
        # The main loop will properly save final stats, release video, and trigger analysis
        return False
    
    # Store original frame for display
    display_frame = frame.copy()
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1.0 / (new_frame_time - st.session_state["realtime_prev_time"] + 1e-6)
    st.session_state["realtime_prev_time"] = new_frame_time
    
    # YOLO inference (on smaller image for speed)
    dets = yolo_inference(frame, yolo_model, max_size=640)
    CONF_THR = 0.4
    filtered = [d for d in dets if d[4] >= CONF_THR]
    
    # Build and merge boxes
    raw_bag = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 0]
    raw_punch = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 5]
    raw_kick = [tuple(map(int, d[:4])) for d in filtered if int(d[5]) == 2]
    
    bag_boxes = merge_overlapping_boxes(raw_bag)
    punch_boxes = merge_overlapping_boxes(raw_punch)
    kick_boxes = merge_overlapping_boxes(raw_kick)
    
    # Check overlaps (for counting, don't draw yet)
    ov_punch = False
    for a in punch_boxes:
        for b in bag_boxes:
            if boxes_intersect(a, b):
                ov_punch = True
                break
        if ov_punch:
            break
    
    ov_kickkne = False
    for a in kick_boxes:
        for b in bag_boxes:
            if boxes_intersect(a, b):
                ov_kickkne = True
                break
        if ov_kickkne:
            break
    
    # Event counting logic
    frame_count = st.session_state["realtime_frame_count"]
    counters = st.session_state["realtime_counters"]
    in_event = st.session_state["realtime_in_event"]
    event_start = st.session_state["realtime_event_start"]
    gap_counter = st.session_state["realtime_gap_counter"]
    
    for action, is_over in [('punch', ov_punch), ('kick-knee', ov_kickkne)]:
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
    
    # Prepare frame vector for transformer (one-hot encoding)
    frame_vector = np.zeros(num_classes, dtype=np.float32)
    for x1f, y1f, x2f, y2f, conf, cls in filtered:
        cls_id = int(cls)
        if 0 <= cls_id < num_classes:
            frame_vector[cls_id] = 1.0
    
    # Add to frame buffer and process transformer chunk periodically
    st.session_state["realtime_transformer_counter"] += 1
    if st.session_state["realtime_transformer_counter"] >= transformer_stride:
        st.session_state["frame_buffer"].append(frame_vector)
        
        # Process transformer chunk if model available (less frequently to reduce overhead)
        if transformer_model and len(st.session_state["frame_buffer"]) >= 32:
            # Only process every 4 buffer additions (every 32 frames) to reduce computation overhead
            buffer_count = len(st.session_state["frame_buffer"])
            if buffer_count % 32 == 0:
                actions = process_transformer_chunk(st.session_state["frame_buffer"], transformer_model, num_classes)
                if actions:
                    st.session_state["action_sequence"].extend(actions)
        
        st.session_state["realtime_transformer_counter"] = 0
    
    # Update statistics
    total_hits = counters['punch'] + counters['kick-knee']
    # Calculate active ratio from actual action frames vs total frames
    action_frames = st.session_state.get("realtime_active_frames", 0)
    total_frames = st.session_state["realtime_frame_count"]
    active_ratio = (action_frames / total_frames) * 100 if total_frames > 0 else 0
    
    st.session_state["current_statistics"] = {
        "punch": counters['punch'],
        "kick-knee": counters['kick-knee'],
        "total_hits": total_hits,
        "active_ratio": active_ratio
    }
    
    # Annotate display frame (draw boxes and text on original frame for display)
    for x1f, y1f, x2f, y2f, conf, cls in filtered:
        cls = int(cls)
        color = {0:(0,255,0), 5:(255,0,255), 2:(0,255,255)}.get(cls, (0,0,255))
        label = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw overlap lines if any
    if ov_punch:
        for a in punch_boxes:
            for b in bag_boxes:
                if boxes_intersect(a, b):
                    ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                    cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                    cv2.line(display_frame, ca, cb, (0,255,255), 2)
    if ov_kickkne:
        for a in kick_boxes:
            for b in bag_boxes:
                if boxes_intersect(a, b):
                    ca = ((a[0]+a[2])//2, (a[1]+a[3])//2)
                    cb = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                    cv2.line(display_frame, ca, cb, (0,255,255), 2)
    
    # Annotate statistics on frame
    cv2.putText(display_frame, f"Hits: {total_hits}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Punch: {counters['punch']}", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(display_frame, f"Kick-Knee: {counters['kick-knee']}", (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Draw FPS
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = display_frame.shape[1] - text_w - margin
    y = margin + text_h
    cv2.putText(display_frame, text, (x, y), font, font_scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    
    # Always update display for smooth streaming (removed frame skipping)
    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    # Only show active ratio if video is not active (stopped)
    active_ratio_display = ""
    if not st.session_state.get("realtime_active", True):
        active_ratio_display = f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: var(--text-secondary);">⚡ Active Ratio:</span>
                <span style="font-weight: 600; color: var(--text-primary);">{active_ratio:.1f}%</span>
            </div>
    """
    
    # Update statistics display with modern card styling
    stats_text = f"""
    <div class="stats-display">
        <h4 style="margin-top: 0; color: var(--text-primary); margin-bottom: 15px;">📊 Real-Time Statistics</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
            <div style="background: rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid var(--success);">
                <div style="color: var(--text-secondary); font-size: 0.85em;">Total Hits</div>
                <div style="font-size: 2em; font-weight: 700; color: var(--success);">{total_hits}</div>
            </div>
            <div style="background: rgba(59, 130, 246, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid var(--info);">
                <div style="color: var(--text-secondary); font-size: 0.85em;">FPS</div>
                <div style="font-size: 2em; font-weight: 700; color: var(--info);">{fps:.1f}</div>
            </div>
        </div>
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: var(--text-secondary);">🥊 Punches:</span>
                <span style="font-weight: 600; color: var(--text-primary);">{counters['punch']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: var(--text-secondary);">🦵 Kicks:</span>
                <span style="font-weight: 600; color: var(--text-primary);">{counters['kick-knee']}</span>
            </div>
            {active_ratio_display}
        </div>
    </div>
    """
    stats_placeholder.markdown(stats_text, unsafe_allow_html=True)
    
    # Don't update graph during video playback - graph will be generated only at end
    # Graph placeholder should already show initial message from initialization
    # This prevents flickering during video processing
    
    # Don't update llm_placeholder here - let main function handle it based on video state
    # This prevents conflicts and ensures AI analysis only shows after video stops
    
    st.session_state["realtime_frame_count"] += 1
    return True

# Execute YOLO model on a Video File
def model_execution():
    st.title("▶️ Model Execution")
    
    # Default Data.YAML; Shouldnt need to be changed without Project Changes.
    default_yaml_path = os.path.join("dataset", "data.yaml")
    yaml_file_path = default_yaml_path
    yolo_classes = load_classes_from_yaml(yaml_file_path)
    yolo_models = [f for f in os.listdir('models') if f.endswith('.pt')]
    selected_yolo_model = st.selectbox("Select YOLO Model", yolo_models)
    model_path = os.path.join('models', selected_yolo_model)
    video_file = st.file_uploader("Upload video", type=['mp4','avi','mov'])

    if video_file:
        video_path = os.path.join("data", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
    if st.button("▶️ Run Models", type="primary") and video_file:
        model = load_yolo_model(model_path)
        if model is None:
            st.error("Failed to load YOLO model.")
            return
        detections = process_video(video_path, model)
        if not detections:
            st.error("No detections.")
            return
        detected_class_ids = []
        all_dets = []
        for frame_idx, frame in enumerate(detections):
            frame_detections = []
            for d in frame:
                if len(d) >= 6:
                    class_id = int(d[5])
                    if 0 <= class_id < len(yolo_classes):
                        detected_class_ids.append(class_id)
                        frame_detections.append([frame_idx, *d[:6]])
            all_dets.append(frame_detections)
        
        # Save CSV & State Success
        csv_file_path = os.path.join("runs", f"yolo_predictions_{video_file.name}.csv")
        df_detections = pd.DataFrame([det for frame in all_dets for det in frame], columns=['frame','x1','y1','x2','y2','confidence','class_id'])
        df_detections.to_csv(csv_file_path, index=False)

        st.success(f"Predictions => {csv_file_path}")

        # Derive a proper .mp4 filename
        input_name    = video_file.name
        base, _       = os.path.splitext(input_name)
        output_name   = f"{base}.mp4"
        out_path      = os.path.join("runs", output_name)

        # Serve it with the correct MIME‐type
        with open(out_path, "rb") as video_file_handle:
            video_bytes = video_file_handle.read()
            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name=os.path.basename(out_path),
                mime="video/mp4"  # or "video/quicktime" for .mov
            )

### Transformer Functions
# Class Instantiation for Transformers Dataset
class TransformerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Class Instantiation for Transformers Model
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

# Prepares Transformer Inputs from YOLO detections CSV
def prepare_transformer_inputs_from_csv(csv_file_or_df, sequence_length=None, stride=None, num_classes=None):
    DEFAULT_SEQ_LEN = 32
    if sequence_length is None:
        sequence_length = DEFAULT_SEQ_LEN
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive int, got {sequence_length!r}")

    if stride is None:
        # stride = sequence_length // 2
        stride = max(4, sequence_length // 4)  # 25% overlap instead of 50%
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

# Function to Train Transformer
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

# Loads the Transformer Model (with caching)
@st.cache_resource
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
        model_.eval()  # Set to eval mode for faster inference
        
        # Use half precision on GPU if available
        if device.type == 'cuda':
            try:
                model_ = model_.half()
            except:
                pass
        
        return model_
    except:
        return None

# Process transformer chunk from frame buffer
def process_transformer_chunk(frame_buffer, transformer_model, num_classes=6):
    """
    Process a chunk of frames through the transformer model.
    frame_buffer: deque or list of frame vectors (one-hot encoded class presence per frame)
    Returns: list of predicted actions for the last frame(s) in the chunk
    """
    if len(frame_buffer) < 32:
        return []
    
    try:
        # Convert buffer to tensor format (batch_size=1, seq_len=32, num_classes)
        frame_list = list(frame_buffer)
        seq_data = np.array(frame_list[-32:], dtype=np.float32)  # Take last 32 frames
        
        # Use appropriate dtype based on model precision
        if device.type == 'cuda' and next(transformer_model.parameters()).dtype == torch.float16:
            seq_tensor = torch.tensor(seq_data, dtype=torch.float16).unsqueeze(0).to(device)
        else:
            seq_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Run inference with optimized mode
        with torch.inference_mode():
            logits = transformer_model(seq_tensor)
        
        # Process predictions for the last frame in the sequence
        probs = torch.sigmoid(logits[0, -1, :]).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        # Map to actions
        class_map = {0: "boxing-bag", 1: "high-guard", 2: "kick-knee",
                     3: "low-guard", 4: "person", 5: "punch"}
        
        # Determine action based on predictions
        action = "idle"
        if preds[5] and preds[0]:  # punch and bag
            action = "punch"
        elif preds[2] and preds[0]:  # kick and bag
            action = "kick-knee"
        elif preds[1]:  # high-guard
            action = "high-guard"
        
        return [action]
    except Exception as e:
        return []

# Interface to train Transformers
def transformer_training_interface():
    st.title("🧠 Transformer Training")
    info_card("📊 Training Data", "Upload CSV file with frame-level detections for transformer model training")
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
        if st.button("🚀 Train Transformer", type="primary"):
            try:
                seq_len = 32
                # stride  = seq_len // 2
                stride = max(4, seq_len // 4)  # 25% overlap instead of 50%
                num_classes = 6
                inputs, labels = prepare_transformer_inputs_from_csv(df_, sequence_length=seq_len, num_classes=num_classes, stride=stride)
                st.write(f"inputs={inputs.shape}, labels={labels.shape}")
                model_ = train_transformer_model(inputs, labels, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=num_classes, num_epochs=100, batch_size=16)
                st.success("Transformer trained.")
            except Exception as e:
                st.error(f"Error training: {e}")

### HMM Functions
# HMM Function to Prepare Sequences for Dashboard
def compress_state_sequence(actions_sequence):
    """
    Merge consecutive repeated states into a single instance.
    E.g. ["idle","idle","punch","punch","high-guard","idle","idle"]
         => ["idle","punch","high-guard","idle"]
    """
    if not actions_sequence:
        return []
    compressed = [actions_sequence[0]]
    for i in range(1, len(actions_sequence)):
        if actions_sequence[i] != compressed[-1]:
            compressed.append(actions_sequence[i])
    return compressed

# Create basic metrics display that works with minimal data
def create_basic_metrics_display(action_sequence, counters, statistics, graph_placeholder):
    """
    Create and display combat sports metrics/charts even with minimal data.
    Works when action_sequence has < 2 items (insufficient for transition graph).
    """
    try:
        # Count actions from sequence
        action_counts = {}
        for action in action_sequence:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Also include counts from counters dict
        if counters:
            for action_key in ['punch', 'kick-knee']:
                if action_key in counters and counters[action_key] > 0:
                    action_counts[action_key] = action_counts.get(action_key, 0) + counters[action_key]
        
        # Get statistics
        total_hits = statistics.get('total_hits', 0) if statistics else 0
        punches = statistics.get('punch', 0) if statistics else (counters.get('punch', 0) if counters else 0)
        kicks = statistics.get('kick-knee', 0) if statistics else (counters.get('kick-knee', 0) if counters else 0)
        active_ratio = statistics.get('active_ratio', 0) if statistics else 0
        
        # Create display HTML with metrics
        metrics_html = f"""
        <div style="background: var(--card-bg); padding: 20px; border-radius: 12px; margin: 10px 0;">
            <h4 style="margin-top: 0; color: var(--text-primary); margin-bottom: 15px;">📊 Combat Sports Metrics</h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px;">
                <div style="background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #EF4444;">
                    <div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 5px;">🥊 Punches</div>
                    <div style="font-size: 2em; font-weight: 700; color: #EF4444;">{punches}</div>
                </div>
                <div style="background: rgba(59, 130, 246, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #3B82F6;">
                    <div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 5px;">🦵 Kicks</div>
                    <div style="font-size: 2em; font-weight: 700; color: #3B82F6;">{kicks}</div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #10B981;">
                    <div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 5px;">🎯 Total Hits</div>
                    <div style="font-size: 2em; font-weight: 700; color: #10B981;">{total_hits}</div>
                </div>
                <div style="background: rgba(139, 92, 246, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #8B5CF6;">
                    <div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 5px;">⚡ Active Ratio</div>
                    <div style="font-size: 2em; font-weight: 700; color: #8B5CF6;">{active_ratio:.1f}%</div>
                </div>
            </div>
        """
        
        # Add action count bar chart if we have action data
        if action_counts:
            # Create pandas DataFrame for bar chart
            action_df = pd.DataFrame({
                'Action': list(action_counts.keys()),
                'Count': list(action_counts.values())
            })
            
            # Display bar chart using Streamlit
            graph_placeholder.markdown(metrics_html, unsafe_allow_html=True)
            graph_placeholder.markdown("### Action Distribution")
            graph_placeholder.bar_chart(action_df.set_index('Action'))
        else:
            # Show metrics even without action data
            metrics_html += """
                <div style="color: var(--text-secondary); margin-top: 15px; font-style: italic;">
                    Start your training session to see action analytics.
                </div>
            """
            graph_placeholder.markdown(metrics_html, unsafe_allow_html=True)
            
        return True
    except Exception as e:
        # Fallback: show basic message
        error_msg = f"Metrics display error: {str(e)}"
        print(f"[Metrics] {error_msg}")
        graph_placeholder.markdown(f"""
        <div style="background: var(--card-bg); padding: 20px; border-radius: 12px;">
            <h4 style="color: var(--text-primary);">📊 Combat Sports Metrics</h4>
            <p style="color: var(--text-secondary);">Metrics will appear here once data is available.</p>
        </div>
        """, unsafe_allow_html=True)
        return False

# Create actions timeline graph - shows actions over time (works with single action)
def create_actions_timeline_graph(action_sequence, graph_placeholder):
    """
    Create a timeline graph showing actions over time/sequence position.
    Works even with just one action.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        if not action_sequence or len(action_sequence) == 0:
            return False
        
        # Create mapping of action types to y-positions and colors
        unique_actions = list(set(action_sequence))
        action_colors = {
            'punch': '#EF4444',
            'kick-knee': '#3B82F6',
            'high-guard': '#10B981',
            'low-guard': '#F59E0B',
            'idle': '#6B7280',
            'boxing-bag': '#8B5CF6'
        }
        
        # Map actions to y-positions (for visualization clarity)
        action_y_pos = {}
        for i, action in enumerate(unique_actions):
            action_y_pos[action] = i
        
        # Prepare data for plotting
        x_values = []  # Sequence indices (time/frame positions)
        y_values = []  # Action type positions
        colors_list = []
        labels_list = []
        
        for idx, action in enumerate(action_sequence):
            x_values.append(idx)
            y_values.append(action_y_pos[action])
            colors_list.append(action_colors.get(action, '#667eea'))
            labels_list.append(action)
        
        # Create figure with dark theme
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='#2a2a3e')
        ax.set_facecolor('#2a2a3e')
        
        # Plot actions over time
        # Use scatter plot for better visibility
        scatter = ax.scatter(x_values, y_values, c=colors_list, s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
        
        # If we have multiple points, draw lines connecting them
        if len(x_values) > 1:
            # Group by action type and draw lines for each action type
            for action in unique_actions:
                action_indices = [i for i, a in enumerate(action_sequence) if a == action]
                if len(action_indices) > 1:
                    action_x = [x_values[i] for i in action_indices]
                    action_y = [action_y_pos[action] for _ in action_indices]
                    ax.plot(action_x, action_y, 
                           color=action_colors.get(action, '#667eea'), 
                           alpha=0.3, linewidth=2)
        
        # Set y-axis labels to action names
        ax.set_yticks(range(len(unique_actions)))
        ax.set_yticklabels(unique_actions, color='white', fontsize=10)
        ax.set_xlabel('Sequence Position (Time)', color='#a0a0a0', fontsize=11)
        ax.set_ylabel('Action Type', color='#a0a0a0', fontsize=11)
        ax.set_title('Actions Over Time', color='white', fontsize=14, fontweight='bold', pad=15)
        
        # Create legend
        legend_patches = []
        for action in unique_actions:
            color = action_colors.get(action, '#667eea')
            count = action_sequence.count(action)
            legend_patches.append(mpatches.Patch(color=color, label=f'{action} ({count})'))
        
        ax.legend(handles=legend_patches, loc='upper right', facecolor='#2a2a3e', 
                 edgecolor='#667eea', labelcolor='white', fontsize=9)
        
        # Style the plot
        ax.grid(True, alpha=0.2, color='#667eea', linestyle='--')
        ax.spines['bottom'].set_color('#667eea')
        ax.spines['top'].set_color('#2a2a3e')
        ax.spines['right'].set_color('#2a2a3e')
        ax.spines['left'].set_color('#667eea')
        ax.tick_params(colors='#a0a0a0')
        
        plt.tight_layout()
        
        # Display in placeholder
        graph_placeholder.markdown("### Actions Timeline")
        graph_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        return True
    except Exception as e:
        error_msg = f"Timeline graph error: {str(e)}"
        print(f"[Timeline Graph] {error_msg}")
        return False

# Create action transition graph visualization
def create_action_transition_graph(action_sequence, max_actions=30):
    """
    Create a network graph showing action transitions (Markov model visualization).
    Returns matplotlib figure or None if insufficient data.
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Get recent actions
        recent = action_sequence[-max_actions:] if len(action_sequence) > max_actions else action_sequence
        if len(recent) < 2:
            return None  # Need at least 2 actions for transitions
        
        # Build transition pairs
        transitions = [(recent[i], recent[i+1]) for i in range(len(recent)-1)]
        
        # Count transitions
        transition_counts = {}
        for trans in transitions:
            transition_counts[trans] = transition_counts.get(trans, 0) + 1
        
        # Create directed graph
        G = nx.DiGraph()
        for (from_action, to_action), count in transition_counts.items():
            if G.has_edge(from_action, to_action):
                G[from_action][to_action]['weight'] += count
            else:
                G.add_edge(from_action, to_action, weight=count)
        
        # Layout
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Create figure with dark background
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2a2a3e')
            ax.set_facecolor('#2a2a3e')
            
            # Draw nodes with action-specific colors
            node_colors = {
                'punch': '#EF4444',
                'kick-knee': '#3B82F6',
                'high-guard': '#10B981',
                'low-guard': '#F59E0B',
                'idle': '#6B7280',
                'boxing-bag': '#8B5CF6'
            }
            colors = [node_colors.get(node, '#667eea') for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, 
                                   node_size=1500, alpha=0.9, edgecolors='white', linewidths=2)
            
            # Draw edges with weights (thicker = more frequent transitions)
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#667eea', 
                                  width=[w*2 for w in weights], alpha=0.6, arrows=True, 
                                  arrowsize=20, arrowstyle='->')
            
            # Labels
            nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_size=10, font_weight='bold')
            
            # Edge labels (transition counts)
            edge_labels = {(u, v): str(G[u][v]['weight']) for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_color='#a0a0a0', font_size=8)
            
            ax.axis('off')
            plt.tight_layout()
            return fig
        
        return None
    except Exception as e:
        # Return None if graph creation fails (graceful degradation)
        return None

# HMM Builder
def build_and_run_hmm(action_sequence):
    """
    Demonstrates the states that occurred in the sequence and 
    the transitions between consecutive states, without 
    returning numeric probabilities or HMM-specific matrices.
    """
    if not action_sequence:
        return {
            "message": "No actions seen."
        }

    # Get unique states in the order they appear
    # (dict.fromkeys(...) preserves first occurrence order)
    states_observed = list(dict.fromkeys(action_sequence))

    # Build a list of pairwise transitions
    transitions_seen = []
    for i in range(len(action_sequence) - 1):
        current_state = action_sequence[i]
        next_state = action_sequence[i + 1]
        transitions_seen.append((current_state, next_state))

    # Create a simple "->" chain to visualize the entire flow
    flow_of_states = " -> ".join(action_sequence)

    return {
        "message": "Markov demonstration",
        "states_observed": states_observed,
        "flow_of_states": flow_of_states
    }

### Results Dashboard
# Select CSV File & Transformer Weights
def transformer_results_dashboard():
    st.title("📊 Transformer Results Dashboard")
    info_card("📈 Analysis Results", "Select your transformer model and CSV file to view detailed statistics and AI analysis")
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
    
    #if st.button("Generate Stats & GPT Summary"):
    csv_path = os.path.join("runs", sel_csv)
    df_ = pd.read_csv(csv_path)
    run_transformer_statistics(os.path.join("models", sel_t), df_, sel_csv)

def build_segments(hit_set, action, min_len, gap_tol, total_frames):
    segments   = []
    in_event   = False
    gap_count  = 0
    start_f    = 0

    for f in range(total_frames):
        is_hit = f in hit_set

        if is_hit:
            gap_count = 0
            if not in_event:
                in_event = True
                start_f  = f
        elif in_event:
            gap_count += 1
            if gap_count >= gap_tol:               
                dur = f - start_f                
                if dur >= min_len:
                    segments.append({
                        'action': action,
                        'start':  start_f,
                        'end':    f - gap_count,
                        'length': dur
                    })
                in_event  = False
                gap_count = 0

    # close event that lasts to EOF
    if in_event and (total_frames - start_f) >= min_len:
        segments.append({
            'action': action,
            'start':  start_f,
            'end':    total_frames - 1,
            'length': total_frames - start_f
        })
    return segments

# Run Results Inference on CSV file produced by YOLO models
def run_transformer_statistics(model_path, df_detections, video_path=None):
    """
    Transformer-based approach that:
      - Infers punch/kick/high-guard presence from frame-level predictions
      - Computes dynamic grace periods from data for hit counting
      - Builds a multi-frame actions_sequence for Markov analysis
      - Compresses consecutive repeats before HMM analysis
      - Reports detailed statistics and distribution metrics
    """
    import matplotlib.pyplot as plt

    # 1) Load Transformer model
    model_ = load_transformer_model(model_path, d_model=64, nhead=2, num_layers=2, 
                                    dim_feedforward=128, dropout=0.1, num_classes=6)
    if not model_:
        st.error("Transformer model not found")
        return

    # 2) Prepare inputs with optimal parameters
    seq_len = 32
    stride = max(4, seq_len // 4)  # 25% overlap
    # stride = seq_len // 2
    inputs, _ = prepare_transformer_inputs_from_csv(df_detections, seq_len, stride, 6)
    if inputs.size == 0:
        st.warning("No valid sequences")
        return

    # 3) Run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_.to(device).eval()
    
    # Ensure inputs match model dtype to prevent dtype mismatch errors
    model_dtype = next(model_.parameters()).dtype
    with torch.no_grad():
        logits = torch.cat([model_(inputs[i].unsqueeze(0).to(dtype=model_dtype).to(device)).cpu() 
                            for i in range(inputs.shape[0])], dim=0)

    # 4) Process predictions
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)  # More sensitive threshold

    # 5) Aggregate predictions
    global_preds = {}
    global_probs = {}

    for win_idx in range(preds.shape[0]):
        start = win_idx * stride
        for t in range(seq_len):
            frame    = start + t
            arr_pred = preds[win_idx, t]
            arr_prob = probs[win_idx, t]

            if frame not in global_preds:
                global_preds[frame] = np.zeros(6, dtype=int)
                global_probs[frame] = np.zeros(6, dtype=float)  # ← init here

            # take the max over overlapping windows
            global_preds[frame] = np.maximum(global_preds[frame], arr_pred)
            global_probs[frame] = np.maximum(global_probs[frame], arr_prob)  # ← update here

    frames = sorted(global_preds.keys())
    if not frames:
        st.warning("No frames predicted.")
        return
    total_frames = frames[-1] + 1

    # 6) Map classes
    class_map = {0: "boxing-bag", 1: "high-guard", 2: "kick-knee",
                 3: "low-guard", 4: "person", 5: "punch"}

    # 7) Build frame-by-frame action sequence
    sequence = []
    
    # 8) Keep track of which frames contain hits
    punch_frames = set()
    kick_frames  = set()

    for f in frames:
        g = global_preds[f]
        if g[5] and g[0]:
            sequence.append("punch")
            # ← INSERT THIS: record punch frames
            punch_frames.add(f)
        elif g[2] and g[0]:
            sequence.append("kick-knee")
            # ← INSERT THIS: record kick frames
            kick_frames.add(f)
        elif g[1]:
            sequence.append("high-guard")
        else:
            sequence.append("idle")

    df_hits = pd.DataFrame({'frame': frames})
    df_hits['punches'] = df_hits['frame'].isin(punch_frames).cumsum()
    df_hits['kicks']   = df_hits['frame'].isin(kick_frames).cumsum()

    df_seq = pd.DataFrame({'frame': frames, 'action': sequence})

    # 9) Segment lengths for each action type
    df_seq['segment'] = (df_seq['action'] != df_seq['action'].shift()).cumsum()
    segs = df_seq.groupby('segment').agg(
        action=('action', 'first'),
        start=('frame', 'first'),
        end=('frame', 'last')
    ).reset_index(drop=True)
    segs['length'] = segs['end'] - segs['start'] + 1

    # 10) Compute durations and ratios
    durations = df_seq['action'].value_counts().reindex(class_map.values(), fill_value=0)
    active = df_seq[df_seq['action'].isin(['punch','kick-knee'])].shape[0]
    active_ratio = active / total_frames * 100

    MIN_LEN   = {'punch': 2, 'kick-knee': 6}
    GAP_TOL   = {'punch': 1, 'kick-knee': 4}

    punch_segments = pd.DataFrame(
        build_segments(punch_frames, 'punch',
                    MIN_LEN['punch'], GAP_TOL['punch'], total_frames)
    )
    kick_segments  = pd.DataFrame(
        build_segments(kick_frames, 'kick-knee',
                    MIN_LEN['kick-knee'], GAP_TOL['kick-knee'], total_frames)
    )

    # 11) Display results
    st.write("### Activity")
    gpt_punch = len(punch_segments)
    gpt_kick = len(kick_segments)
    st.write(f"Estimated Punches: {gpt_punch}")
    st.write(f"Estimated Kicks: {gpt_kick}")
    st.write(f"Active: {active_ratio:.1f}%")
    st.write(f"Resting: {100 - active_ratio:.1f}%")
    

    # 12) Stats over segments
    stats = segs.groupby('action')['length'].describe().rename(columns={
        'count':'Segments', 'mean':'AvgLen', 'min':'MinLen', '25%':'Q1','50%':'Median','75%':'Q3','max':'MaxLen'
    })

    # 13) HMM analysis on compressed sequence
    compressed = compress_state_sequence(sequence)

    # 14) Transition counts
    trans = pd.DataFrame({'prev': compressed[:-1], 'next': compressed[1:]})
    trans_mat = pd.crosstab(trans['prev'], trans['next']).reindex(index=stats.index, columns=stats.index, fill_value=0)

    # 15) GPT Testing
    from ollama import chat

    # 1. Break out your instructions into a system message
    system_prompt = """
    You are an expert combat-sports analyst. Talk like a human.
    Only use the statistics provided; do not invent or infer any others.  
    The opponent is always a stationary boxing bag.  
    Produce exactly five observations:  
    • Each one sentence, ≤15 words.  
    • Clearly numbered 1–5.  
    Do not add any extra text or explanations.
    """

    # 2. Build your user content separately
    user_stats = {
        "Punches": gpt_punch,
        "Kicks":   gpt_kick,
        "Action Sequences": compressed
    }
    user_prompt = f"Here are the statistics:\n{user_stats}"

    # 3. Call chat once and stream the response
    stream = chat(
        model="gpt-oss:120b-cloud",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        stream=True,
    )

    # 4. Stream and display
    st.write("### LLAMA 3.2 Response")
    full = ""
    placeholder = st.empty()

    for chunk in stream:
        # Allow stopping mid-stream
        if st.session_state.get("stop_requested", False):
            placeholder.markdown(full + "\n\n*⏹️ Generation stopped.*")
            break

        # Append new content and update
        delta = chunk["message"]["content"]
        full += delta
        placeholder.markdown(full)  

    st.info("🧠 Generated…")

    # 16) Build a true instance‐count per action
    instances = {}

    # 17) Use your filtered segments for punch & kick
    instances['punch']     = len(punch_segments)
    instances['kick-knee'] = len(kick_segments)

    # 18) For all other actions, just count segments in segs
    for action in ['high-guard', 'low-guard', 'idle']:
        instances[action] = segs[segs['action'] == action].shape[0]

    # 19) turn into a DataFrame
    instance_counts = (
        pd.Series(instances)
        .rename_axis('Action')
        .to_frame(name='Instances')
    )

    # 20) Show Charts & CSV preview
    st.write("### Action Instances Chart")
    st.bar_chart(instance_counts)

    st.write(f"### CSV Preview - {video_path}")
    st.write(df_detections.head(5))

    # 21) Action Timeline (Gantt)
    from matplotlib.patches import Patch
    st.write("### Action Timeline Diagram")
    fig_timeline, ax = plt.subplots(figsize=(8, 2))
    color_map = {'punch': 'red', 'kick-knee': 'blue', 'high-guard': 'green', 'idle': 'gray'}

    # 22) Draw each segment
    for _, row in segs.iterrows():
        ax.barh(0,
                row['length'],
                left=row['start'],
                color=color_map[row['action']],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5)

    # 23) Add grid for frame ticks
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    handles = [Patch(color=col, label=act) for act, col in color_map.items()]
    ax.legend(handles=handles,
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.2),
            loc='upper center',
            frameon=False)

    ax.set_yticks([])
    ax.set_xlabel('Frame')
    st.pyplot(fig_timeline, use_container_width=False)

    # 24) Segment-Length Histogram
    st.write("### Segment-Length Histogram")
    fig_hist, ax = plt.subplots()
    for act, grp in segs.groupby('action'):
        ax.hist(grp['length'], bins=10, alpha=0.5, label=act)
    ax.set_xlabel('Segment Length (frames)'); ax.set_ylabel('Count'); ax.legend()
    st.pyplot(fig_hist, use_container_width=False)

    # 25) Cumulative Hits Over Time
    st.write("### Cumulative Hits Over Time")
    df_hits = pd.DataFrame({'frame': frames})
    df_hits['punches'] = df_hits['frame'].isin(punch_frames).cumsum()
    df_hits['kicks']   = df_hits['frame'].isin(kick_frames).cumsum()
    fig_cum_hits, ax = plt.subplots()
    ax.plot(df_hits['frame'], df_hits['punches'], label='Punches')
    ax.plot(df_hits['frame'], df_hits['kicks'],   label='Kicks')
    ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Hits'); ax.legend()
    st.pyplot(fig_cum_hits, use_container_width=False)

    st.write("### Inter-Hit Interval Distribution (Event-based)")

    # 1) Extract event start‐frames, if any
    if not punch_segments.empty and 'start' in punch_segments:
        punch_events = np.sort(punch_segments['start'].values)
    else:
        punch_events = np.array([])

    if not kick_segments.empty and 'start' in kick_segments:
        kick_events = np.sort(kick_segments['start'].values)
    else:
        kick_events = np.array([])

    # 2) Compute intervals between successive hit instances
    intervals_p = np.diff(punch_events) if punch_events.size > 1 else np.array([])
    intervals_k = np.diff(kick_events)  if kick_events.size  > 1 else np.array([])

    # 3) Plot
    fig_intervals, ax = plt.subplots()
    if intervals_p.size:
        ax.hist(intervals_p, bins=10, alpha=0.5, label='Punch Intervals')
    if intervals_k.size:
        ax.hist(intervals_k, bins=10, alpha=0.5, label='Kick Intervals')

    ax.set_xlabel('Frames Between Hit Instances')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig_intervals, use_container_width=False)

    st.write("### Transition Probability Heatmap")
    st.write("### Transition Matrix")
    # create a Styler with a blue gradient and integer formatting
    styled = (
        trans_mat
        .style
        .format("{:.0f}")                        # no decimals
        .background_gradient(cmap="Blues")       # color scale
        .set_properties(**{
            "border": "1px solid black",         # grid lines
            "text-align": "center",
            "font-family": "monospace"
        })
    )

    # 26) Streamlit will render the styled DataFrame
    st.write(styled)

    # 27) Transition Probability Heatmap
    compressed_seq = compress_state_sequence(sequence)
    trans = pd.DataFrame({'prev': compressed_seq[:-1], 'next': compressed_seq[1:]})
    trans_mat = pd.crosstab(trans['prev'], trans['next'])
    prob_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).fillna(0)
    fig_heatmap, ax = plt.subplots()
    cax = ax.matshow(prob_mat.values, cmap='Blues')
    fig_heatmap.colorbar(cax)
    ax.set_xticks(range(len(prob_mat.columns))); ax.set_xticklabels(prob_mat.columns, rotation=45)
    ax.set_yticks(range(len(prob_mat.index)));    ax.set_yticklabels(prob_mat.index)
    ax.set_title('Transition Probabilities')
    st.pyplot(fig_heatmap, use_container_width=False)

    # 28) Max Class Confidence Over Time
    st.write("### Max Class Confidence Over Time")
    df_conf = pd.DataFrame({
        'frame': frames,
        'max_conf': [global_probs[f].max() for f in frames]
    })
    fig_conf, ax = plt.subplots()
    ax.plot(df_conf['frame'], df_conf['max_conf'])
    ax.hlines(0.8, xmin=0, xmax=total_frames, linestyles='dashed')
    ax.set_xlabel('Frame'); ax.set_ylabel('Max Confidence')
    st.pyplot(fig_conf, use_container_width=False)

    # 29) State Transition Network
    st.write("### State Transition Network")
    import networkx as nx  # make sure this is at the top of your file

    G = nx.DiGraph()
    for (u, v), w in prob_mat.stack().items():
        if w > 0:
            G.add_edge(u, v, weight=w)

    fig_network, ax = plt.subplots(figsize=(7, 7))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, ax=ax)

    # Build edge_labels separately to avoid inline f-string issues
    edge_labels = { (u, v): f"{d['weight']:.2f}" 
                    for (u, v), d in G.edges.items() }

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax
    )
    st.pyplot(fig_network, use_container_width=False)

    # 30) Download All Results
    import io, zipfile

    # 1) Gather your figures into a dict with updated file names
    charts = {
        "01_action_timeline.png":              fig_timeline,
        "02_segment_length_histogram.png":     fig_hist,
        "03_cumulative_hits.png":              fig_cum_hits,
        "04_inter_hit_intervals.png":          fig_intervals,
        "05_transition_heatmap.png":           fig_heatmap,
        "06_max_confidence_over_time.png":     fig_conf,
        "07_state_transition_network.png":     fig_network
    }

    # 2) Build an in-memory ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w") as zf:
        # save each chart PNG
        for name, fig in charts.items():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            zf.writestr(name, buf.read())
        # save key tables as CSVs
        zf.writestr("segments.csv",              segs.to_csv(index=False))
        zf.writestr("transition_counts.csv",     trans_mat.to_csv())
        zf.writestr("transition_probabilities.csv", prob_mat.to_csv())
        zf.writestr("durations.csv",             durations.to_frame("DurationFrames").to_csv())
        # optional: raw sequence
        zf.writestr("action_sequence.txt",       "\n".join(sequence))

    zip_buffer.seek(0)

    # 3) Offer the ZIP for download
    st.download_button(
        label="📥 Download All Results",
        data=zip_buffer.read(),
        file_name="transformer_results_bundle.zip",
        mime="application/zip"
    )

# Check if Ollama is available and running, and optionally check model availability
def check_ollama_available(model_name="gpt-oss:120b-cloud"):
    """
    Check if Ollama service is available and running, and optionally verify model availability.
    Returns (is_available: bool, error_message: str) tuple.
    
    Args:
        model_name: Optional model name to check availability (default: "gpt-oss:120b-cloud")
    """
    try:
        import requests
        
        # First, check if Ollama service is running
        try:
            # Test connection to Ollama API
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code != 200:
                error_msg = f"Ollama service returned status {response.status_code}"
                print(f"[Ollama] ERROR: {error_msg}")
                return False, error_msg
            
            print("[Ollama] Service is available and responding")
            
            # Optionally check if the specific model is available
            if model_name:
                try:
                    models_data = response.json()
                    available_models = [model.get('name', '') for model in models_data.get('models', [])]
                    
                    # Check if model is available (handle variations like 'gpt-oss:120b-cloud')
                    model_found = False
                    for avail_model in available_models:
                        if model_name in avail_model or avail_model in model_name:
                            model_found = True
                            print(f"[Ollama] Model '{avail_model}' found (matched '{model_name}')")
                            break
                    
                    if not model_found:
                        warning_msg = f"Model '{model_name}' not found in available models. Available: {', '.join(available_models[:5])}"
                        print(f"[Ollama] WARNING: {warning_msg}")
                        # Don't fail, just warn - model might be pullable or name might vary
                except Exception as model_check_error:
                    print(f"[Ollama] WARNING: Could not check model availability: {model_check_error}")
                    # Continue anyway - connection is working
            
            return True, ""
            
        except requests.exceptions.ConnectionError:
            error_msg = "Ollama service is not running. Please start Ollama locally (run 'ollama serve' in command prompt)."
            print(f"[Ollama] ERROR: {error_msg}")
            return False, error_msg
        except requests.exceptions.Timeout:
            error_msg = "Ollama service timed out. Please check if Ollama is running (run 'ollama serve' in command prompt)."
            print(f"[Ollama] ERROR: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Ollama connection error: {str(e)}"
            print(f"[Ollama] ERROR: {error_msg}")
            return False, error_msg
            
    except ImportError:
        error_msg = "Requests package not installed. Please install with: pip install requests"
        print(f"[Ollama] ERROR: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Failed to check Ollama availability: {str(e)}"
        print(f"[Ollama] ERROR: {error_msg}")
        return False, error_msg

# Validate that a specific Ollama model is actually callable
def validate_ollama_model(model_name="gpt-oss:120b-cloud", timeout=5):
    """
    Test if a specific Ollama model is actually available and callable.
    Returns (is_valid: bool, error_message: str) tuple.
    
    Args:
        model_name: Name of the model to validate
        timeout: Timeout in seconds for the test call
    """
    try:
        from ollama import chat
        
        print(f"[Model Validation] Testing model: {model_name}")
        
        # Attempt a minimal test call to the model
        try:
            test_response = chat(
                model=model_name,
                messages=[
                    {"role": "user", "content": "test"}
                ],
                stream=False,
                options={"num_predict": 1}  # Very short response for testing
            )
            
            # Check if we got a valid response
            if test_response and 'message' in test_response:
                print(f"[Model Validation] Model '{model_name}' is valid and responding")
                return True, ""
            else:
                error_msg = f"Model '{model_name}' returned invalid response format"
                print(f"[Model Validation] ERROR: {error_msg}")
                return False, error_msg
                
        except Exception as model_error:
            error_msg = f"Model '{model_name}' is not available or failed: {str(model_error)}"
            print(f"[Model Validation] ERROR: {error_msg}")
            return False, error_msg
            
    except ImportError:
        error_msg = "Ollama package not installed. Please install with: pip install ollama"
        print(f"[Model Validation] ERROR: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Failed to validate model: {str(e)}"
        print(f"[Model Validation] ERROR: {error_msg}")
        return False, error_msg

# Real-time LLM analysis function
def get_realtime_llm_analysis(statistics, action_sequence):
    """
    Get LLM interpretation of current real-time statistics.
    Updates every 1 second with current stats.
    """
    # Check Ollama availability first
    ollama_available, ollama_error = check_ollama_available()
    if not ollama_available:
        return f"LLM unavailable: {ollama_error}"
    
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
        
        # Get recent action sequence (last 20 actions)
        recent_sequence = action_sequence[-20:] if len(action_sequence) >= 20 else action_sequence
        compressed = compress_state_sequence(recent_sequence)
        
        user_stats = {
            "Punches": statistics.get("punch", 0),
            "Kicks": statistics.get("kick-knee", 0),
            "Total Hits": statistics.get("total_hits", 0),
            "Active Ratio": f"{statistics.get('active_ratio', 0):.1f}%",
            "Recent Actions": compressed[-10:] if len(compressed) >= 10 else compressed
        }
        user_prompt = f"Here are the current real-time statistics:\n{user_stats}"
        
        # Call chat (non-streaming for real-time updates)
        response = chat(
            model="gpt-oss:120b-cloud",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        
        return response["message"]["content"]
    except Exception as e:
        return f"LLM analysis unavailable: {str(e)}"

# End-of-video AI analysis function with streaming
def generate_end_of_video_analysis(final_stats, action_sequence, analysis_queue):
    """
    Generate AI analysis at end of video with streaming chunks.
    Uses queue to pass data to main thread (no direct session state access).
    analysis_queue: Queue object for thread-safe communication
    """
    start_time = time.time()
    timeout_seconds = 60  # 60 second timeout
    
    print(f"[Analysis] Starting end-of-video analysis generation")
    
    # Check Ollama availability first
    ollama_available, ollama_error = check_ollama_available()
    if not ollama_available:
        error_msg = f"Analysis error: {ollama_error}"
        print(f"[Analysis] {error_msg}")
        try:
            analysis_queue.put({"type": "complete", "data": error_msg})
        except Exception as queue_error:
            print(f"[Analysis] Failed to send error to queue: {queue_error}")
        return error_msg
    
    # Validate model is actually callable
    model_name = "gpt-oss:120b-cloud"
    model_valid, model_error = validate_ollama_model(model_name)
    if not model_valid:
        error_msg = f"Analysis error: Model validation failed - {model_error}"
        print(f"[Analysis] {error_msg}")
        try:
            analysis_queue.put({"type": "complete", "data": error_msg})
        except Exception as queue_error:
            print(f"[Analysis] Failed to send error to queue: {queue_error}")
        return error_msg
    
    print(f"[Analysis] Model validated, proceeding with analysis")
    
    # Ensure completion message is ALWAYS sent via try-finally
    completion_sent = False
    try:
        from ollama import chat
        
        system_prompt = """
        You are an Expert Combat Sports Statistics Analyst.

        ROLE:
        - Deliver concise, data-driven insights on combat-sport performance metrics.
        - Assume the opponent is always a stationary boxing bag.
        - Use only the provided statistics. Do NOT infer, speculate, or add unverified data.

        OUTPUT FORMAT:
        Produce exactly FIVE numbered observations (1–5):
        - Each observation must be ONE sentence (≤15 words).
        - Use professional analytical language (no filler, emotion, or hype).
        - Do NOT include introductions, summaries, or additional commentary.

        Your goal: highlight performance patterns, strengths, weaknesses, or trends visible in the data alone.
        """

        
        compressed = compress_state_sequence(action_sequence)
        
        user_stats = {
            "Punches": final_stats.get("punch", 0),
            "Kicks": final_stats.get("kick-knee", 0),
            "Action Sequences": compressed
        }
        user_prompt = f"Here are the statistics:\n{user_stats}"
        
        # Log input schema to Ollama
        messages_structure = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        print(f"[Ollama Input Schema]")
        print(f"  - Model: 'gpt-oss:120b-cloud'")
        print(f"  - Stream: True")
        print(f"  - Messages structure: {type(messages_structure).__name__} with {len(messages_structure)} items")
        print(f"  - Message[0] (system): role='{messages_structure[0]['role']}', content_length={len(messages_structure[0]['content'])} chars")
        print(f"    System prompt preview: {messages_structure[0]['content'][:100]}...")
        print(f"  - Message[1] (user): role='{messages_structure[1]['role']}', content_length={len(messages_structure[1]['content'])} chars")
        print(f"    User stats: {user_stats}")
        print(f"    User prompt preview: {messages_structure[1]['content'][:150]}...")
        
        # Call chat with streaming enabled
        # Wrap in timeout check - if chat() itself blocks, we can't detect it here,
        # but the UI timeout (60s) will catch it and mark as failed
        stream = None
        try:
            stream = chat(
                model="gpt-oss:120b-cloud",
                messages=messages_structure,
                stream=True,
            )
        except ConnectionError as conn_error:
            # Connection-specific error
            error_msg = f"Analysis error: Cannot connect to Ollama. Please ensure Ollama is running (try 'ollama serve' in terminal). Details: {str(conn_error)}"
            print(f"[Analysis] Connection error: {conn_error}")
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
            except:
                pass
            return error_msg
        except Exception as chat_error:
            # Generic error with detailed message
            error_type = type(chat_error).__name__
            error_details = str(chat_error)
            
            # Provide helpful messages for common errors
            if "connection refused" in error_details.lower() or "cannot connect" in error_details.lower():
                error_msg = f"Analysis error: Ollama is not running. Please start Ollama with 'ollama serve' and ensure the model 'gpt-oss:120b-cloud' is available."
            elif "model" in error_details.lower() and "not found" in error_details.lower():
                error_msg = f"Analysis error: Model 'gpt-oss:120b-cloud' not found. Please pull the model with 'ollama pull gpt-oss:120b-cloud'."
            elif "timeout" in error_details.lower():
                error_msg = f"Analysis error: Connection to Ollama timed out. Check if the model is loaded and Ollama is responding."
            else:
                error_msg = f"Analysis error: Failed to connect to AI model ({error_type}). Details: {error_details}"
            
            print(f"[Analysis] Chat error ({error_type}): {error_details}")
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
            except:
                pass
            return error_msg
        
        # Validate stream was created
        if stream is None:
            error_msg = "Analysis error: No response from AI model."
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
            except:
                pass
            return error_msg
        
        # Validate stream is iterable
        if stream is None:
            error_msg = "Analysis error: Stream object is None"
            print(f"[Analysis] ERROR: {error_msg}")
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
                completion_sent = True
            except Exception as queue_error:
                print(f"[Analysis] Failed to send error to queue: {queue_error}")
            return error_msg
        
        # Check if stream is iterable
        try:
            iter(stream)
        except TypeError:
            error_msg = "Analysis error: Stream is not iterable"
            print(f"[Analysis] ERROR: {error_msg}")
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
                completion_sent = True
            except Exception as queue_error:
                print(f"[Analysis] Failed to send error to queue: {queue_error}")
            return error_msg
        
        # Process stream chunks - use queue instead of session state
        full_text = ""
        chunk_buffer = ""
        last_chunk_time = time.time()
        chunk_interval = 1.0  # Send chunks every 1 second
        chunks_received = 0
        
        print(f"[Analysis] Starting stream iteration")
        
        # Process stream with timeout protection
        try:
            for chunk in stream:
                chunks_received += 1
                
                # Log output schema for first chunk and every 100th chunk to show structure
                if chunks_received == 1 or chunks_received % 100 == 0:
                    print(f"[Ollama Output Schema - Chunk #{chunks_received}]")
                    print(f"  - Chunk type: {type(chunk).__name__}")
                    print(f"  - Chunk keys: {list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")
                    if isinstance(chunk, dict):
                        if 'message' in chunk:
                            print(f"  - Message type: {type(chunk['message']).__name__}")
                            print(f"  - Message keys: {list(chunk['message'].keys()) if isinstance(chunk['message'], dict) else 'N/A'}")
                            if isinstance(chunk['message'], dict) and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                print(f"  - Content type: {type(content).__name__}")
                                print(f"  - Content length: {len(content) if isinstance(content, str) else 'N/A'} chars")
                                if isinstance(content, str) and chunks_received == 1:
                                    print(f"  - Content preview (first chunk): {repr(content[:50])}")
                        # Log any other keys in chunk
                        other_keys = [k for k in chunk.keys() if k != 'message']
                        if other_keys:
                            print(f"  - Other chunk keys: {other_keys}")
                
                # Check for overall timeout on each chunk
                if time.time() - start_time > timeout_seconds:
                    error_msg = "Analysis timeout: The analysis took too long to complete. Please try again."
                    print(f"[Analysis] ERROR: {error_msg} (received {chunks_received} chunks)")
                    try:
                        analysis_queue.put({"type": "complete", "data": error_msg})
                        completion_sent = True
                    except Exception as queue_error:
                        print(f"[Analysis] Failed to send timeout error to queue: {queue_error}")
                    return error_msg
                
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        full_text += content
                        chunk_buffer += content
                        
                        # Send chunk every ~1 second via queue (no session state access)
                        current_time = time.time()
                        if current_time - last_chunk_time >= chunk_interval:
                            # Put chunk in queue for main thread to process
                            try:
                                queue_message = {"type": "chunk", "data": chunk_buffer}
                                analysis_queue.put(queue_message)
                                print(f"[Analysis] Sent chunk ({len(chunk_buffer)} chars, total: {len(full_text)} chars)")
                                if chunks_received <= 3:  # Log queue message schema for first few chunks
                                    print(f"  [Queue Schema] Message: type='{queue_message['type']}', data_length={len(queue_message['data'])} chars")
                            except Exception as queue_error:
                                print(f"[Analysis] WARNING: Failed to send chunk to queue: {queue_error}")
                            chunk_buffer = ""  # Clear buffer
                            last_chunk_time = current_time
                else:
                    print(f"[Analysis] WARNING: Received chunk without expected structure")
                    print(f"  - Chunk keys: {list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")
                    print(f"  - Chunk content: {chunk}")
                    
        except Exception as stream_error:
            # If stream iteration fails, send error
            error_msg = f"Analysis error: Stream processing failed - {str(stream_error)}"
            print(f"[Analysis] ERROR: {error_msg} (received {chunks_received} chunks before error)")
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
                completion_sent = True
            except Exception as queue_error:
                print(f"[Analysis] Failed to send stream error to queue: {queue_error}")
            return error_msg
        
        print(f"[Analysis] Stream iteration completed. Total chunks: {chunks_received}, Total text length: {len(full_text)}")
        
        # Add final chunk if any remaining
        if chunk_buffer:
            try:
                final_chunk_message = {"type": "chunk", "data": chunk_buffer}
                analysis_queue.put(final_chunk_message)
                print(f"[Analysis] Sent final chunk buffer ({len(chunk_buffer)} chars)")
                print(f"  [Queue Schema] Final chunk: type='{final_chunk_message['type']}', data_length={len(final_chunk_message['data'])} chars")
            except Exception as queue_error:
                print(f"[Analysis] WARNING: Failed to send final chunk: {queue_error}")
        
        # Store final result via queue - ensure this always happens
        if full_text:
            try:
                completion_message = {"type": "complete", "data": full_text}
                print(f"[Ollama Output Schema - Complete]")
                print(f"  - Queue message type: '{completion_message['type']}'")
                print(f"  - Final text length: {len(full_text)} chars")
                print(f"  - Final text preview: {full_text[:100]}...")
                print(f"  - Queue message structure: dict with keys: {list(completion_message.keys())}")
                print(f"  - Message format: {{'type': '{completion_message['type']}', 'data': str({len(completion_message['data'])} chars)}}")
                
                analysis_queue.put(completion_message)
                completion_sent = True
                print(f"[Analysis] Sent completion signal with {len(full_text)} chars of text")
            except Exception as queue_error:
                print(f"[Analysis] ERROR: Failed to send completion to queue: {queue_error}")
        else:
            # If no text was received, send error
            error_msg = "Analysis incomplete: No response received from AI model."
            print(f"[Analysis] ERROR: {error_msg} (received {chunks_received} chunks)")
            try:
                error_completion_message = {"type": "complete", "data": error_msg}
                print(f"[Ollama Output Schema - Complete (Error)]")
                print(f"  - Queue message type: '{error_completion_message['type']}'")
                print(f"  - Error message: {error_msg}")
                print(f"  - Queue message structure: dict with keys: {list(error_completion_message.keys())}")
                
                analysis_queue.put(error_completion_message)
                completion_sent = True
            except Exception as queue_error:
                print(f"[Analysis] Failed to send incomplete error to queue: {queue_error}")
        
        return full_text if full_text else "Analysis incomplete."
    except ImportError as e:
        error_msg = f"Analysis error: Ollama not available - {str(e)}"
        print(f"[Analysis] ERROR: {error_msg}")
        if not completion_sent:
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
                completion_sent = True
            except Exception as queue_error:
                print(f"[Analysis] Failed to send ImportError to queue: {queue_error}")
        return error_msg
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        print(f"[Analysis] ERROR: Unexpected exception - {error_msg}")
        import traceback
        print(f"[Analysis] Traceback: {traceback.format_exc()}")
        if not completion_sent:
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
                completion_sent = True
            except Exception as queue_error:
                print(f"[Analysis] Failed to send exception to queue: {queue_error}")
        return error_msg
    finally:
        # Ensure completion is always sent, even if something went wrong
        if not completion_sent:
            error_msg = "Analysis error: Unexpected failure - no completion message sent"
            print(f"[Analysis] CRITICAL: {error_msg}")
            try:
                analysis_queue.put({"type": "complete", "data": error_msg})
            except Exception as queue_error:
                print(f"[Analysis] CRITICAL: Failed to send final error to queue: {queue_error}")

# Centralized cleanup function for video processing state
def cleanup_video_processing(keep_stats=False):
    """
    Centralized cleanup for video processing state.
    
    Args:
        keep_stats: If True, preserve statistics and action_sequence for viewing after stop.
                   If False, clear everything (used when uploading new video).
    """
    # Always release video source (idempotent - safe to call multiple times)
    video_source = st.session_state.get("realtime_video_source")
    if video_source is not None:
        try:
            # Check if video source is still open before releasing
            if hasattr(video_source, 'isOpened') and video_source.isOpened():
                video_source.release()
        except Exception as e:
            # Ignore errors if already released or invalid
            pass
        finally:
            st.session_state["realtime_video_source"] = None
    
    # Always reset active state and camera type
    st.session_state["realtime_active"] = False
    if "camera_type" in st.session_state:
        del st.session_state["camera_type"]
    
    # Only clear statistics and sequences if explicitly requested (new video upload)
    if not keep_stats:
        st.session_state["current_statistics"] = {"punch": 0, "kick-knee": 0, "total_hits": 0, "active_ratio": 0}
        st.session_state["action_sequence"] = []
        st.session_state["frame_buffer"].clear()
        st.session_state["realtime_counters"] = {'punch': 0, 'kick-knee': 0}
        st.session_state["realtime_frame_count"] = 0
        st.session_state["realtime_active_frames"] = 0
        st.session_state["realtime_gap_counter"] = {'punch': 0, 'kick-knee': 0}
        st.session_state["realtime_in_event"] = {'punch': False, 'kick-knee': False}
        st.session_state["realtime_event_start"] = {'punch': 0, 'kick-knee': 0}
        st.session_state["session_final_stats"] = None
        st.session_state["analysis_generating"] = False
        st.session_state["analysis_stream_chunks"] = []
        st.session_state["llm_last_result"] = "AI Analysis will appear here..."
        st.session_state["analysis_final_displayed"] = False

# Helper function to poll analysis queue and update session state
def poll_analysis_queue():
    """Poll the analysis queue from background thread and update session state in main thread."""
    if "analysis_chunks_queue" not in st.session_state:
        return False

    queue = st.session_state["analysis_chunks_queue"]
    updated = False

    while True:
        try:
            item = queue.get_nowait()
            if item["type"] == "chunk":
                if "analysis_stream_chunks" not in st.session_state:
                    st.session_state["analysis_stream_chunks"] = []
                st.session_state["analysis_stream_chunks"].append(item["data"])
                updated = True
            elif item["type"] == "complete":
                st.session_state["llm_last_result"] = item["data"]
                st.session_state["analysis_generating"] = False
                # Don't clear chunks here - let them persist until final result is displayed
                # Chunks will be cleared in display logic when final result is ready
                updated = True
        except Exception as e:
            if type(e).__name__ != "Empty":  # Only log non-empty exceptions
                print(f"[Analysis] Error polling queue: {e}")
            break

    return updated

# Real-time Analysis Main Interface
def realtime_analysis():
    st.title("⏱️Combat Sports Analyst")
    
    # Auto-select best YOLO model
    yolo_model_path = select_best_yolo_model()
    if not yolo_model_path:
        st.error("No YOLO models found in 'models' directory. Please train or add a model first.")
        return
    
    # Define transformer model path before status cards (available throughout function)
    transformer_model_path = "models/transformer_model.pth"
    ollama_model_name = "gpt-oss:120b-cloud"  # Ollama model name
    
    # Load YOLO model (already optimized and cached)
    yolo_model = load_yolo_model(yolo_model_path)
    if yolo_model is None:
        st.error("Failed to load YOLO model.")
        return
    
    # Try to load transformer model
    transformer_model = None
    if os.path.exists(transformer_model_path):
        transformer_model = load_transformer_model(transformer_model_path)
        if not transformer_model:
            st.warning("⚠️ Transformer model file exists but failed to load. Continuing without transformer analysis.")
    else:
        st.info("💡 No transformer model found. YOLO detection will work, but transformer-based action recognition will be disabled.")
    
    # Video source selection - only file uploads supported
    # Removed browser camera option - only file uploads are supported
    
    video_path = None
    
    # Upload Video File section (only supported video source)
    # Track current uploaded file to detect changes (upload or deletion)
    if "current_uploaded_file_name" not in st.session_state:
        st.session_state["current_uploaded_file_name"] = None
    
    uploaded_file = st.file_uploader("📤 Upload video file", type=['mp4', 'avi', 'mov'])
    
    # Detect file changes (new upload or deletion)
    current_file_name = uploaded_file.name if uploaded_file else None
    file_changed = current_file_name != st.session_state["current_uploaded_file_name"]
    
    if uploaded_file:
        video_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)
        
        # Only write file if it's a new upload (not already in session state)
        if file_changed:
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            st.session_state["current_uploaded_file_name"] = uploaded_file.name
            # Stop any active processing and clear ALL data when new file is uploaded
            cleanup_video_processing(keep_stats=False)
    elif file_changed and st.session_state["current_uploaded_file_name"]:
        # File was deleted/removed
        st.session_state["current_uploaded_file_name"] = None
        # Stop any active processing and clear data when file is removed
        cleanup_video_processing(keep_stats=False)
    
    # Only proceed if we have a valid uploaded file
    if uploaded_file:
        video_path = os.path.join("data", uploaded_file.name)
        
        # Show Start/Stop buttons side by side
        if st.session_state.get("realtime_active", False) and st.session_state.get("camera_type") == "uploaded_video":
            # Show Stop button when active - centered
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                if st.button("⏹️ Stop Analysis", use_container_width=True, type="primary"):
                    # Collect final statistics before stopping
                    final_stats = {
                        "punch": st.session_state["realtime_counters"].get('punch', 0),
                        "kick-knee": st.session_state["realtime_counters"].get('kick-knee', 0),
                        "total_hits": st.session_state["current_statistics"].get('total_hits', 0),
                    }
                    # Recalculate active ratio from action frames
                    action_frames = st.session_state.get("realtime_active_frames", 0)
                    total_frames = st.session_state["realtime_frame_count"]
                    final_stats["active_ratio"] = (action_frames / total_frames) * 100 if total_frames > 0 else 0
                    action_sequence = st.session_state.get("action_sequence", [])
                    
                    # Store final stats before cleanup
                    st.session_state["session_final_stats"] = final_stats
                    
                    # Stop video processing but KEEP statistics visible
                    cleanup_video_processing(keep_stats=True)
                    
                    # Trigger rerun to show final stats and analysis
                    st.rerun()
        else:
            # Show Start button when not active - centered
            btn_start_col1, btn_start_col2, btn_start_col3 = st.columns([1, 2, 1])
            with btn_start_col2:
                if st.button("▶️ Start Analysis", use_container_width=True, type="primary"):
                    video_source = cv2.VideoCapture(video_path)
                    if not video_source.isOpened():
                        st.error(f"❌ Failed to open video: {video_path}")
                        return
                    
                    # Reset all counters and statistics when starting fresh
                    st.session_state["current_statistics"] = {"punch": 0, "kick-knee": 0, "total_hits": 0, "active_ratio": 0}
                    st.session_state["action_sequence"] = []
                    st.session_state["frame_buffer"].clear()
                    st.session_state["realtime_counters"] = {'punch': 0, 'kick-knee': 0}
                    st.session_state["realtime_frame_count"] = 0
                    st.session_state["realtime_gap_counter"] = {'punch': 0, 'kick-knee': 0}
                    st.session_state["realtime_in_event"] = {'punch': False, 'kick-knee': False}
                    st.session_state["realtime_event_start"] = {'punch': 0, 'kick-knee': 0}
                    st.session_state["session_final_stats"] = None
                    st.session_state["analysis_generating"] = False
                    st.session_state["analysis_stream_chunks"] = []
                    st.session_state["llm_last_result"] = "AI Analysis will appear here..."
                    st.session_state["analysis_final_displayed"] = False
                    st.session_state["realtime_active_frames"] = 0
                    st.session_state["frames_processed_this_chunk"] = 0  # Reset chunk counter
                    
                    # Set video source and activate processing
                    st.session_state["realtime_video_source"] = video_source
                    st.session_state["realtime_active"] = True
                    st.session_state["camera_type"] = "uploaded_video"
                    # Initialize rerun timestamp
                    st.session_state["last_pc_rerun"] = time.time()
                    # DO NOT rerun - processing loop will start immediately in same execution
                    # Page will only rerun when user uploads new file or deletes file
    
    # Create vertical layout and process frames (for uploaded video files)
    if st.session_state.get("realtime_active", False) and st.session_state.get("realtime_video_source"):
        # Always create placeholders fresh on each rerun
        # This ensures they remain valid after reruns
        st.markdown("---")
        # Create vertical layout - full width stacked rows
        # Video feed (full width)
        video_placeholder = st.empty()
        
        # Statistics (full width, below video)
        st.markdown("---")
        stats_placeholder = st.empty()
        
        # Graph and AI Analysis removed from active section - will show in stopped video section after completion
        
        # Process frames in chunks to avoid Streamlit timeouts for long videos
        try:
            # Initialize frame processing state if not exists
            if "frames_processed_this_chunk" not in st.session_state:
                st.session_state["frames_processed_this_chunk"] = 0
            
            frames_processed = st.session_state["frames_processed_this_chunk"]
            video_ended = False
            max_frames_per_chunk = 30  # Process 30 frames per execution to avoid timeouts
            
            # Process frames in chunks - prevents blocking for long videos
            frames_this_chunk = 0
            while (st.session_state.get("realtime_active", False) and 
                   st.session_state.get("realtime_video_source") and
                   frames_this_chunk < max_frames_per_chunk):
                if not process_realtime_frame(
                    yolo_model, 
                    transformer_model, 
                    video_placeholder, 
                    stats_placeholder, 
                    None,  # graph_placeholder removed
                    None   # llm_placeholder removed
                ):
                    # Video ended naturally
                    video_ended = True
                    st.session_state["realtime_active"] = False
                    break
                
                frames_processed += 1
                frames_this_chunk += 1
                
                # Control FPS with sleep
                time.sleep(0.033)  # ~30 FPS target
                
                # Poll queue periodically (every 30 frames = ~1 second) for analysis updates
                if frames_processed % 30 == 0:
                    poll_analysis_queue()
            
            # Update session state with current frame count
            st.session_state["frames_processed_this_chunk"] = frames_processed
            
            # Poll queue one final time
            poll_analysis_queue()
            
            # If video is still active and we haven't reached end, trigger rerun to continue processing
            if (st.session_state.get("realtime_active", False) and 
                st.session_state.get("realtime_video_source") and 
                not video_ended):
                # Continue processing in next execution
                st.rerun()
            
            # When video ends naturally, prepare for analysis but don't display anything yet
            if video_ended:
                # Collect final statistics
                final_stats = {
                    "punch": st.session_state["realtime_counters"].get('punch', 0),
                    "kick-knee": st.session_state["realtime_counters"].get('kick-knee', 0),
                    "total_hits": st.session_state["current_statistics"].get('total_hits', 0),
                }
                # Recalculate active ratio from action frames
                action_frames = st.session_state.get("realtime_active_frames", 0)
                total_frames = st.session_state["realtime_frame_count"]
                final_stats["active_ratio"] = (action_frames / total_frames) * 100 if total_frames > 0 else 0
                action_sequence = st.session_state.get("action_sequence", [])
                
                # Store final stats
                st.session_state["session_final_stats"] = final_stats
                
                # Reset chunk counter
                st.session_state["frames_processed_this_chunk"] = 0
                
                # Release video source using centralized cleanup (idempotent)
                cleanup_video_processing(keep_stats=True)
                
                print("[Video] Video source released before showing results")
                
                # Trigger rerun - stopped video section will handle display
                st.rerun()
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            cleanup_video_processing(keep_stats=True)
    
    # Display final stats when video is stopped but data is preserved
    elif uploaded_file and not st.session_state.get("realtime_active", False):
        # Check if we have preserved data to display
        if st.session_state.get("session_final_stats") or st.session_state.get("action_sequence"):
            st.markdown("---")
            st.info("📊 Video analysis complete. Results displayed below.")
            
            # Create placeholders for displaying preserved data
            stats_display = st.empty()
            graph_display = st.empty()
            llm_display = st.empty()
            
            # Display statistics
            final_stats = st.session_state.get("session_final_stats", {})
            counters = st.session_state.get("realtime_counters", {})
            if final_stats or counters:
                punch_count = final_stats.get('punch', 0) or counters.get('punch', 0)
                kick_count = final_stats.get('kick-knee', 0) or counters.get('kick-knee', 0)
                total_hits = final_stats.get('total_hits', 0) or (punch_count + kick_count)
                active_ratio = final_stats.get('active_ratio', 0)
                
                stats_html = f"""
                <div class="stats-display">
                    <h3 style="margin-top: 0; color: var(--text-primary); margin-bottom: 20px;">📈 Final Statistics</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                        <div style="background: rgba(239, 68, 68, 0.2); padding: 20px; border-radius: 12px; border-left: 4px solid #EF4444;">
                            <div style="color: var(--text-secondary); font-size: 0.9em; margin-bottom: 8px;">🥊 Punches</div>
                            <div style="font-size: 2.5em; font-weight: 700; color: #EF4444;">{punch_count}</div>
                        </div>
                        <div style="background: rgba(59, 130, 246, 0.2); padding: 20px; border-radius: 12px; border-left: 4px solid #3B82F6;">
                            <div style="color: var(--text-secondary); font-size: 0.9em; margin-bottom: 8px;">🦵 Kicks/Knees</div>
                            <div style="font-size: 2.5em; font-weight: 700; color: #3B82F6;">{kick_count}</div>
                        </div>
                        <div style="background: rgba(16, 185, 129, 0.2); padding: 20px; border-radius: 12px; border-left: 4px solid #10B981;">
                            <div style="color: var(--text-secondary); font-size: 0.9em; margin-bottom: 8px;">🎯 Total Hits</div>
                            <div style="font-size: 2.5em; font-weight: 700; color: #10B981;">{total_hits}</div>
                        </div>
                    </div>
                </div>
                """
                stats_display.markdown(stats_html, unsafe_allow_html=True)
            
            # Display action graph if available
            action_seq = st.session_state.get("action_sequence", [])
            if len(action_seq) >= 2:
                graph_fig = create_action_transition_graph(action_seq, max_actions=30)
                if graph_fig:
                    graph_display.markdown("### 📊 Action Transitions")
                    graph_display.pyplot(graph_fig, use_container_width=True)
                    import matplotlib.pyplot as plt
                    plt.close(graph_fig)
            
            # Display AI analysis - using working dashboard code approach
            action_seq = st.session_state.get("action_sequence", [])
            
            # Only show analysis if we have stats to analyze
            if final_stats or counters or action_seq:
                try:
                    from ollama import chat
                    
                    # Prepare statistics for AI (same as dashboard)
                    punch_count = final_stats.get('punch', 0) or counters.get('punch', 0)
                    kick_count = final_stats.get('kick-knee', 0) or counters.get('kick-knee', 0)
                    compressed = compress_state_sequence(action_seq) if action_seq else []
                    
                    # 1. Break out your instructions into a system message
                    system_prompt = """
                    ROLE:
                    You are an expert combat-sports analyst writing for a coaching audience.

                    TASK:
                    Analyze the supplied session statistics to identify tactical behaviors and decisions demonstrated by the boxer.
                    Summarise these in exactly five concise tactical observations.

                    RULES:
                    1. Use only the provided data — no external knowledge or fabrication.
                    2. Assume the opponent is a stationary heavy bag.
                    3. Each observation must describe a *tactical principle* or *decision*, not just a physical action.
                    4. Each sentence must be ≤18 words and numbered 1–5.
                    5. Output absolutely nothing except those five numbered lines.

                    STYLE:
                    - Begin every sentence with “The boxer…” to maintain lexical consistency.
                    - Use action-oriented tactical language (e.g., manages distance, controls tempo, conserves energy, adjusts output).
                    - Keep tone analytical and precise, suitable for a performance coach.

                    TACTICAL FRAMEWORK (implicit guidance):
                    Consider elements such as:
                    • Range control and positioning  
                    • Pressure and timing  
                    • Defense and endurance management  
                    • Adaptation or response to fatigue indicators  
                    • Balance of power vs. accuracy

                    EXAMPLE OUTPUT:
                    1. The boxer manages range effectively by sustaining consistent jab frequency at mid-distance.  
                    2. The boxer controls tempo with short output bursts followed by measured recovery phases.  
                    3. The boxer adjusts punch selection to conserve energy as intensity increases.  
                    4. The boxer prioritises accuracy over volume, suggesting tactical pacing awareness.  
                    5. The boxer overall demonstrates disciplined control and adaptive shot sequencing.
                    """
                    
                    # 2. Build your user content separately
                    user_stats = {
                        "Punches": punch_count,
                        "Kicks": kick_count,
                        "Action Sequences": compressed
                    }
                    user_prompt = f"Here are the statistics:\n{user_stats}"
                    
                    # 3. Call chat once and stream the response
                    stream = chat(
                        model="gpt-oss:120b-cloud",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        stream=True,
                    )
                    
                    # 4. Stream and display
                    llm_display.markdown("### 🤖 AI Analysis")
                    full = ""
                    placeholder = llm_display.empty()
                    
                    for chunk in stream:
                        # Allow stopping mid-stream
                        if st.session_state.get("stop_requested", False):
                            placeholder.markdown(full + "\n\n*⏹️ Generation stopped.*")
                            break
                        
                        # Append new content and update
                        delta = chunk["message"]["content"]
                        full += delta
                        placeholder.markdown(full)
                    
                    st.info("🧠 Generated…")
                except ImportError:
                    llm_display.markdown("""
                    <div class="ai-analysis-card">
                        <h4 style="margin-top: 0; color: var(--text-primary); margin-bottom: 15px;">🤖 AI Analysis</h4>
                        <div style="color: var(--error);">❌ Ollama not installed. Install with: <code>pip install ollama</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    error_msg = str(e)
                    if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                        llm_display.markdown(f"""
                        <div class="ai-analysis-card">
                            <h4 style="margin-top: 0; color: var(--text-primary); margin-bottom: 15px;">🤖 AI Analysis</h4>
                            <div style="color: var(--error); line-height: 1.6;">
                                ❌ Cannot connect to Ollama.<br><br>
                                <strong>Please ensure:</strong><br>
                                1. Ollama is running (run <code>ollama serve</code> in terminal)<br>
                                2. Model is available (run <code>ollama pull gpt-oss:120b-cloud</code>)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        llm_display.markdown(f"""
                        <div class="ai-analysis-card">
                            <h4 style="margin-top: 0; color: var(--text-primary); margin-bottom: 15px;">🤖 AI Analysis</h4>
                            <div style="color: var(--error);">❌ Analysis error: {error_msg}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Download processed video section
            uploaded_file_name = st.session_state.get("current_uploaded_file_name")
            if uploaded_file_name and yolo_model:
                st.markdown("---")
                download_col1, download_col2 = st.columns([2, 1])
                
                with download_col1:
                    st.markdown("### 📥 Download Processed Video")
                    st.markdown("Download the video with YOLO detections and bounding boxes drawn.")
                
                with download_col2:
                    # Check if processed video already exists
                    video_base_name = os.path.splitext(uploaded_file_name)[0]
                    processed_video_path = os.path.join("runs", f"{video_base_name}.mp4")
                    
                    # Ensure runs directory exists
                    os.makedirs("runs", exist_ok=True)
                    
                    # Process video if it doesn't exist yet
                    if not os.path.exists(processed_video_path):
                        if st.button("🎬 Generate & Download", use_container_width=True, type="primary"):
                            with st.spinner("Processing video with YOLO detections..."):
                                try:
                                    # Get the video path
                                    video_path = os.path.join("data", uploaded_file_name)
                                    if os.path.exists(video_path):
                                        # Use existing process_video function to generate processed video
                                        detections = process_video(video_path, yolo_model)
                                        if detections:
                                            st.success("✅ Video processed successfully!")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to process video.")
                                    else:
                                        st.error(f"❌ Original video not found: {video_path}")
                                except Exception as e:
                                    st.error(f"❌ Error processing video: {str(e)}")
                    else:
                        # Video already exists, offer download
                        try:
                            with open(processed_video_path, "rb") as video_file:
                                video_bytes = video_file.read()
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=video_bytes,
                                    file_name=f"{video_base_name}_processed.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                    type="primary"
                                )
                        except Exception as e:
                            st.error(f"❌ Error reading processed video: {str(e)}")

### Run Main & Check Internet
def main():
    # Directly load real-time analysis - no sidebar needed
    realtime_analysis()

if __name__ == "__main__":
    check_internet_connection()
    main()