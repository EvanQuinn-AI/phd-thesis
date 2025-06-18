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

# Error Handling
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
torch.classes.__path__ = []   # neutralize the broken proxy
warnings.filterwarnings("ignore", category=FutureWarning)

# Check if Device is CUDA ready
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("CUDA: ", torch.version.cuda, torch.cuda.is_available())

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
    # Links to labeling tools
    st.markdown(
        """
        ### Labeling Tools
        - [Roboflow](https://roboflow.com/)
        - [LabelImg](https://github.com/tzutalin/labelImg)
        - [CVAT](https://github.com/openvinotoolkit/cvat)
        """
    )

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

# Loads YOLOv5 from torch hub
@st.cache_resource
def load_yolo_model(weights='models/best.pt'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Function to train new YOLO models
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

# Runs YOLO inference and returns Detections 
def yolo_inference(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model.iou = 0.2
    model.conf = 0.1
    results = model(rgb_frame)
    try:
        detections = results.xyxy[0].cpu().numpy()
    except:
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
        filtered = [d for d in dets if d[4] >= CONF_THR]

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
            label = class_names[cls]
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
                cv2.putText(frame,f"{class_names[cls]} {conf:.2f}",
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

# Execute YOLO model on a Video File
def model_execution():
    st.title("Model Execution")
    
    # Default Data.YAML; Shouldnt need to be changed without Project Changes.
    default_yaml_path = os.path.join("dataset", "data.yaml")
    yaml_file_path = default_yaml_path # Comment This Line Out of You Need To Upload Data.YAML
    
    # yaml_file_path = None
    # if os.path.exists(default_yaml_path):
    #     yaml_file_path = default_yaml_path
    # uploaded_yaml_file = st.file_uploader("Select data.yaml file", type=["yaml"])
    
    # if uploaded_yaml_file is not None:
    #     yaml_file_path = os.path.join("dataset", uploaded_yaml_file.name)
    #     with open(yaml_file_path, "wb") as f:
    #         f.write(uploaded_yaml_file.getbuffer())
    #     st.success(f"Uploaded YAML: {uploaded_yaml_file.name}")
    # if yaml_file_path is None:
    #     st.warning("Please upload data.yaml first.")
    #     return

    yolo_classes = load_classes_from_yaml(yaml_file_path)
    yolo_models = [f for f in os.listdir('models') if f.endswith('.pt')]
    selected_yolo_model = st.selectbox("Select YOLO Model", yolo_models)
    model_path = os.path.join('models', selected_yolo_model)
    video_file = st.file_uploader("Upload video", type=['mp4','avi','mov'])

    if video_file:
        video_path = os.path.join("data", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
    if st.button("Run Models") and video_file:
        model = load_yolo_model(model_path)
        model = model.to(device)
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
        with open(out_path, "rb") as video_file:
            video_bytes = video_file.read()
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

# Loads the Transformer Model
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

# Interface to train Transformers
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
    
    #if st.button("Generate Stats & GPT Summary"):
    csv_path = os.path.join("runs", sel_csv)
    df_ = pd.read_csv(csv_path)
    run_transformer_statistics(os.path.join("models", sel_t), df_, sel_csv)

def build_segments(hit_set, action,
                   min_len, gap_tol, total_frames):
    segments   = []
    in_event   = False
    gap_count  = 0
    start_f    = 0          # placeholder

    for f in range(total_frames):
        is_hit = f in hit_set

        if is_hit:
            gap_count = 0
            if not in_event:
                in_event = True
                start_f  = f
        elif in_event:
            gap_count += 1
            if gap_count >= gap_tol:               # **identical to online**
                dur = f - start_f                  # same length formula
                if dur >= min_len:
                    segments.append({
                        'action': action,
                        'start':  start_f,
                        'end':    f - gap_count,   # last *hit* frame
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
    with torch.no_grad():
        logits = torch.cat([model_(inputs[i].unsqueeze(0).to(device)).cpu() 
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
        model="gemma3:4b",
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