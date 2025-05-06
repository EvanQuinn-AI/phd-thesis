import os
import shutil
import streamlit as st
import sys
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
import yaml
import glob
import subprocess
import pathlib
import warnings
import socket
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Dict
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

#########################
#      PRE-SETUP
#########################

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
pathlib.PosixPath = pathlib.WindowsPath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)

# Ensure essential dirs
for d in ['data','models','runs']:
    os.makedirs(d, exist_ok=True)

# Add local yolov5
yolov5_path = os.path.abspath("yolov5")
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# Map: 0=boxing-bag, 1=cross, 2=high-guard, 3=hook, 4=kick, 5=low-guard, 6=person
CLASS_MAP = {
    0: "boxing-bag",
    1: "cross",
    2: "high-guard",
    3: "hook",
    4: "kick",
    5: "low-guard",
    6: "person"
}

session_defaults = {
    'dataset_folder': None,
    'data_yaml_path': None,
    'yolo_classes': []
}

for k,v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

#########################
# INTERNET CHECK
#########################
def check_internet_connection():
    try:
        socket.create_connection(("www.google.com",80), timeout=5)
        return True
    except OSError:
        return False

def download_yolov5():
    if not os.path.exists('yolov5'):
        print("YOLOv5 repository not found. Checking for Git installation...")
        git_exists = shutil.which("git")
        if git_exists is None:
            st.error("Git is not installed or not found in PATH. Please install Git to continue.")
            return False
        print("Git found. Downloading YOLOv5...")
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'], check=True)
            st.success("YOLOv5 repository downloaded successfully.")
        except subprocess.CalledProcessError:
            st.error("Failed to download YOLOv5. Please ensure Git is installed and accessible.")
            return False
    else:
        print("YOLOv5 repository already exists.")
    return True

#########################
# YAML CLASSES
#########################
def load_classes_from_yaml(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8', errors='replace') as file:
            data = yaml.safe_load(file)
            class_names = data.get('names', [])
            if class_names:
                st.session_state['yolo_classes'] = class_names
            else:
                print("No class names found in YAML file.")
            return class_names
    except Exception as e:
        st.error(f"Failed to load classes from YAML: {e}")
        return []

def update_yaml_paths(yaml_path, dataset_folder):
    if not yaml_path or not dataset_folder:
        raise ValueError("yaml_path and dataset_folder must be valid paths")

    try:
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        # Update train/val/test in the YAML
        yaml_data['train'] = os.path.join(dataset_folder, 'train/images').replace('\\', '/')
        yaml_data['val']   = os.path.join(dataset_folder, 'valid/images').replace('\\', '/')
        yaml_data['test']  = os.path.join(dataset_folder, 'test/images').replace('\\', '/')

        with open(yaml_path, 'w') as file:
            yaml.dump(yaml_data, file)

        st.success(f"Successfully updated paths in {yaml_path}")
    except Exception as e:
        st.error(f"An error occurred while updating the YAML file: {e}")

#########################
# TIME-SERIES TRANSFORMER
#########################
class ActionRecognitionTransformer(nn.Module):
    def __init__(
        self,
        input_size=7,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=7
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer_encoder(emb)
        logits = self.fc(out)
        return logits

#########################
# PREPARE INPUTS FROM CSV
#########################
def prepare_transformer_inputs_from_csv(
    csv_file_or_df,
    sequence_length=20,
    num_classes=7,
    stride=10
):
    """
    Reads a CSV with columns [frame, class_id].
    Produces an (N, sequence_length, num_classes) tensor for inputs
    and an identical shape for labels (multi-label: 1 if class present, else 0).
    """
    if isinstance(csv_file_or_df, pd.DataFrame):
        df = csv_file_or_df
    else:
        df = pd.read_csv(csv_file_or_df)

    if 'frame' not in df.columns or 'class_id' not in df.columns:
        raise ValueError("CSV must have 'frame' and 'class_id' columns.")

    grouped = df.groupby('frame')
    frames = sorted(grouped.groups.keys())

    max_frame = max(frames) if frames else 0
    frame_vectors = [[0]*num_classes for _ in range(max_frame+1)]
    for f_idx in frames:
        class_ids_this_frame = grouped.get_group(f_idx)['class_id'].to_list()
        for cid in class_ids_this_frame:
            cid = int(cid)
            if 0 <= cid < num_classes:
                # Just set to 1 if present
                frame_vectors[f_idx][cid] = 1

    input_chunks = []
    label_chunks = []
    i = 0
    while i + sequence_length <= len(frame_vectors):
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


#########################
# DATASET & LOADER
#########################
class TransformerDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

#########################
# TRAIN TRANSFORMER
#########################
def train_transformer_model(
    inputs, labels,
    d_model=64,
    nhead=2,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    num_classes=7,
    num_epochs=100,
    batch_size=16
):
    """
    Trains the Transformer on the multi-label time series problem.
    Expects inputs & labels of shape (N, seq_len, num_classes).
    Splits data 80/20 for training/validation, uses BCEWithLogitsLoss.
    """
    if isinstance(inputs, list):
        inputs = torch.stack(inputs)
    if isinstance(labels, list):
        labels = torch.stack(labels)

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("inputs shape:", inputs.shape, "labels shape:", labels.shape)

    N = len(inputs)
    if N < 2:
        # If we have fewer than 2 sequences, just train on what we have
        train_in, train_lb = inputs, labels
        val_in,   val_lb   = torch.empty(0), torch.empty(0)
    else:
        split_idx = int(0.8*N)
        train_in  = inputs[:split_idx]
        train_lb  = labels[:split_idx]
        val_in    = inputs[split_idx:]
        val_lb    = labels[split_idx:]

    train_ds = TransformerDataset(train_in, train_lb)
    val_ds   = TransformerDataset(val_in, val_lb)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model_ = ActionRecognitionTransformer(
        input_size=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes
    ).to(device_)

    criterion = nn.BCEWithLogitsLoss().to(device_)
    optimizer = optim.Adam(model_.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    progress_bar = st.progress(0)
    epoch_text   = st.empty()
    best_val_loss= float('inf')
    patience     = 25
    patience_counter = 0

    for epoch in range(num_epochs):
        model_.train()
        total_train_loss = 0.0
        for seqb, lblb in train_loader:
            seqb = seqb.to(device_)
            lblb = lblb.to(device_)
            optimizer.zero_grad()
            outs = model_(seqb)
            # Flatten for BCE
            loss = criterion(outs.view(-1, num_classes), lblb.view(-1, num_classes))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()

        val_loss = 0.0
        model_.eval()
        if len(val_ds) > 0:
            with torch.no_grad():
                for seqb, lblb in val_loader:
                    seqb = seqb.to(device_)
                    lblb = lblb.to(device_)
                    outs = model_(seqb)
                    l    = criterion(outs.view(-1,num_classes), lblb.view(-1,num_classes))
                    val_loss += l.item()

        epoch_text.text(f"Epoch[{epoch+1}/{num_epochs}] "
                        f"TrainLoss:{total_train_loss:.4f} "
                        f"ValLoss:{val_loss:.4f}")
        progress_bar.progress((epoch+1)/num_epochs)

        # Early stopping logic
        if len(val_ds) > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    st.warning("Early stopping triggered!")
                    # Attempt to load last best model (if saved previously)
                    try:
                        model_.load_state_dict(torch.load("models/transformer_model.pth"))
                    except:
                        pass
                    break

    final_path = "models/transformer_model.pth"
    torch.save(model_.state_dict(), final_path)
    print(f"Transformer model saved at {final_path}")
    return model_

######################### 
# YOLO + Execution
#########################
@st.cache_resource
def load_yolo_model(weights='models/best.pt'):
    """
    Loads a YOLOv5 model from the given weights path.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def boxes_intersect(boxA, boxB):
    """
    Return True if two bounding boxes overlap.
    box format: (x1, y1, x2, y2)
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB
    return not (x2A < x1B or x2B < x1A or y2A < y1B or y2B < y1A)

def calculate_iou(boxA, boxB):
    """
    Intersection Over Union for bounding boxes.
    """
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    x_left   = max(xA1, xB1)
    y_top    = max(yA1, yB1)
    x_right  = min(xA2, xB2)
    y_bottom = min(yA2, yB2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left)*(y_bottom - y_top)
    areaA = (xA2 - xA1)*(yA2 - yA1)
    areaB = (xB2 - xB1)*(yB2 - yB1)
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def yolo_inference(frame, model):
    """
    Run YOLO on a single frame, returning detections as Nx6:
    (x1, y1, x2, y2, confidence, class_id).
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    try:
        detections = results.xyxy[0].cpu().numpy()
    except AttributeError:
        st.error("YOLO model did not return the expected format. Check the model's output.")
        detections = []
    return detections

############################################################
# COLOR HISTOGRAM + ID SLOTS (1 and 2)
############################################################
def compute_color_histogram(frame, box, bins=(8,8,8)):
    (x1, y1, x2, y2) = box
    # Ensure the coordinates are in valid range
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0,1,2], None, bins, [0,180,0,256,0,256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def compare_hist(histA, histB):
    """Compute correlation between two histograms (range -1..1)."""
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

############################################################
# TWO FIXED ID TRACKING
############################################################
def update_two_person_ids(frame, person_boxes, tracked):
    """
    Keep exactly 2 ID slots: ID=1 and ID=2.
    Each slot has { 'hist':..., 'box':..., 'action_counts': {...} }.
    We pick the top 2 largest person boxes each frame (by area),
    then match them by histogram correlation to existing ID's hist.
    """
    used_ids = set()

    # Sort by area desc
    def box_area(b):
        x1,y1,x2,y2 = b
        return (x2 - x1)*(y2 - y1)

    person_boxes_sorted = sorted(person_boxes, key=box_area, reverse=True)
    person_boxes_sorted = person_boxes_sorted[:2]  # only keep top 2

    for new_box in person_boxes_sorted:
        new_hist = compute_color_histogram(frame, new_box)
        
        # If ID1 is empty => fill it
        if tracked["1"]["box"] is None and "1" not in used_ids:
            tracked["1"]["box"]  = new_box
            tracked["1"]["hist"] = new_hist
            used_ids.add("1")
            continue
        
        # If ID2 is empty => fill it
        if tracked["2"]["box"] is None and "2" not in used_ids:
            tracked["2"]["box"]  = new_box
            tracked["2"]["hist"] = new_hist
            used_ids.add("2")
            continue

        # Both IDs used => pick best correlation
        if tracked["1"]["box"] is not None:
            corr1 = compare_hist(new_hist, tracked["1"]["hist"])
        else:
            corr1 = -999

        if tracked["2"]["box"] is not None:
            corr2 = compare_hist(new_hist, tracked["2"]["hist"])
        else:
            corr2 = -999

        if corr1 > corr2:
            if "1" not in used_ids:
                tracked["1"]["box"]  = new_box
                tracked["1"]["hist"] = new_hist
                used_ids.add("1")
            else:
                tracked["2"]["box"]  = new_box
                tracked["2"]["hist"] = new_hist
                used_ids.add("2")
        else:
            if "2" not in used_ids:
                tracked["2"]["box"]  = new_box
                tracked["2"]["hist"] = new_hist
                used_ids.add("2")
            else:
                tracked["1"]["box"]  = new_box
                tracked["1"]["hist"] = new_hist
                used_ids.add("1")

    return tracked

############################################################
# MAIN PROCESS VIDEO - TWO PERSON TRACKING
############################################################
def find_action_owner(action_box, tracked):
    """
    The person whose box fully contains the action center
    is the owner. If neither/both boxes contain the center, return None.
    """
    x1, y1, x2, y2 = action_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    def inside_box(cx, cy, box):
        if box is None:
            return False
        bx1, by1, bx2, by2 = box
        return (bx1 <= cx <= bx2) and (by1 <= cy <= by2)

    belongs_to = []
    for pid, pdata in tracked.items():
        if pdata["box"] is None:
            continue
        if inside_box(cx, cy, pdata["box"]):
            belongs_to.append(pid)

    # If exactly one box contains the center
    if len(belongs_to) == 1:
        return belongs_to[0]

    # If none or both => return None, or pick randomly
    return None

def process_video(video_path, model, csv_out_path="runs/detections.csv"):
    """
    Processes a video, tracks two persons (ID=1, ID=2),
    counts each action only if it overlaps with the OTHER person's bounding box,
    uses a grace period, and saves all detections to CSV (with headers).
    
    Returns:
      all_detections (list): 
         - all_detections[frame_idx] is an Nx6 numpy array
           (x1, y1, x2, y2, conf, class_id) for that frame
    """
    class_names = st.session_state.get('yolo_classes', [])
    if len(class_names) < 7:
        class_names = [CLASS_MAP[i] for i in range(7)]

    if not model:
        st.error("No YOLO model loaded.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return []

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base         = os.path.splitext(os.path.basename(video_path))[0]
    out_dir      = "runs"
    os.makedirs(out_dir, exist_ok=True)
    out_video    = os.path.join(out_dir, f"{base}_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (frame_width, frame_height))

    # COLLECT ALL DETECTIONS
    all_detections = []

    # Two IDs, track their actions + last hit frames
    tracked = {
       "1": {
         "box": None,
         "hist": None,
         "action_counts": {
           "cross":0, "hook":0, "kick":0, 
           "high-guard":0, "low-guard":0
         },
         "last_hit_frame": {
            "cross": -1, "hook": -1, "kick": -1,
            "high-guard": -1, "low-guard": -1
         }
       },
       "2": {
         "box": None,
         "hist": None,
         "action_counts": {
           "cross":0, "hook":0, "kick":0, 
           "high-guard":0, "low-guard":0
         },
         "last_hit_frame": {
            "cross": -1, "hook": -1, "kick": -1,
            "high-guard": -1, "low-guard": -1
         }
       }
    }

    # Grace period in frames
    action_grace_period = {
        'cross': 6,     # ~0.2 seconds
        'hook': 8,      # ~0.27 seconds
        'kick': 15,     # ~0.5 seconds
        "high-guard": 6,
        "low-guard": 6
    }

    frame_idx = 0
    progress_bar = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        try:
            dets = results.xyxy[0].cpu().numpy()
        except:
            dets = np.empty((0,6), dtype=np.float32)

        # Store detections for this frame
        all_detections.append(dets)

        # Separate out person vs actions
        person_boxes = []
        action_boxes = {
          "cross": [], "hook": [], "kick": [], 
          "high-guard": [], "low-guard": []
        }

        CONF_THRESH = 0.5
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            if conf < CONF_THRESH:
                continue

            if cls_id == 6:  # person
                person_boxes.append((x1,y1,x2,y2))
                color = (255,0,0)
            elif cls_id == 1:
                action_boxes["cross"].append((x1,y1,x2,y2))
                color = (255,0,255)
            elif cls_id == 3:
                action_boxes["hook"].append((x1,y1,x2,y2))
                color = (255,165,0)
            elif cls_id == 4:
                action_boxes["kick"].append((x1,y1,x2,y2))
                color = (0,255,255)
            elif cls_id == 2:
                action_boxes["high-guard"].append((x1,y1,x2,y2))
                color = (127,255,127)
            elif cls_id == 5:
                action_boxes["low-guard"].append((x1,y1,x2,y2))
                color = (255,255,0)
            else:
                # e.g. boxing-bag or unknown
                color = (0,0,255)

            # Draw the bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label_text = f"{class_names[int(cls_id)]} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update two-person tracking
        tracked = update_two_person_ids(frame, person_boxes, tracked)

        # Draw ID labels
        for pid, pdata in tracked.items():
            if pdata["box"] is not None:
                bx1, by1, bx2, by2 = pdata["box"]
                cv2.putText(frame, f"ID {pid}", (bx1, by1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

        # Assign action ownership by center, check overlap with the other ID
        for act_name, boxes in action_boxes.items():
            for box in boxes:
                owner_id = find_action_owner(box, tracked)

                if owner_id is not None:
                    # The "other" person
                    other_id = "2" if owner_id=="1" else "1"
                    if tracked[other_id]["box"] is not None:
                        if boxes_intersect(box, tracked[other_id]["box"]):
                            # Check grace period
                            last_frame = tracked[owner_id]["last_hit_frame"][act_name]
                            if (frame_idx - last_frame) > action_grace_period[act_name]:
                                tracked[owner_id]["action_counts"][act_name] += 1
                                tracked[owner_id]["last_hit_frame"][act_name] = frame_idx

        writer.write(frame)
        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    # Close video resources
    cap.release()
    writer.release()
    progress_bar.empty()
    st.success(f"Done processing {video_path}.")

    # Print summary
    for pid, pdata in tracked.items():
        st.write(f"**ID {pid}** => {pdata['action_counts']}")

    # -----------------------------
    # SAVE DETECTIONS TO CSV
    # -----------------------------
    rows = []
    # all_detections[i] => detections for frame i
    # shape => Nx6 => [ [x1, y1, x2, y2, conf, cls_id], ... ]
    for frame_i, dets_i in enumerate(all_detections):
        for d in dets_i:
            x1, y1, x2, y2, conf, cid = d
            row = [frame_i, x1, y1, x2, y2, conf, cid]
            rows.append(row)

    df_out = pd.DataFrame(rows,
        columns=["frame","x1","y1","x2","y2","confidence","class_id"])
    df_out.to_csv(csv_out_path, index=False)
    st.success(f"CSV saved => {csv_out_path}")

    return all_detections

#########################
# MODEL EXECUTION
#########################
def model_execution():
    """
    Use the 2-person stable ID approach with overlap-based counting.
    """
    st.title("Model Execution")

    default_yaml_path = os.path.join("dataset","data.yaml")
    yaml_file_path = None
    if os.path.exists(default_yaml_path):
        st.info(f"Default data.yaml found at {default_yaml_path}")
        yaml_file_path = default_yaml_path

    uploaded_yaml_file = st.file_uploader("Select data.yaml file", type=["yaml"])
    if uploaded_yaml_file:
        yaml_file_path = os.path.join("dataset", uploaded_yaml_file.name)
        with open(yaml_file_path,"wb") as f:
            f.write(uploaded_yaml_file.getbuffer())
        st.success(f"Uploaded data.yaml => {uploaded_yaml_file.name}")

    if yaml_file_path is None:
        st.warning("No data.yaml provided. Please upload or provide one.")
        return

    load_classes_from_yaml(yaml_file_path)

    # Select YOLO model from 'models' folder
    available_models = [f for f in os.listdir("models") if f.endswith(".pt")]
    if not available_models:
        st.error("No YOLO .pt models found in 'models' directory.")
        return

    chosen_model = st.selectbox("Select YOLO Model", available_models)
    chosen_model_path = os.path.join("models", chosen_model)

    # Upload a video
    video_file = st.file_uploader("Upload a video", type=["mp4","avi","mov"])
    video_path = None
    if video_file:
        video_path = os.path.join("data", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.success(f"Video uploaded: {video_file.name}")

    if st.button("Run Models") and video_path:
        model = load_yolo_model(chosen_model_path)
        if model is None:
            st.error("Failed to load YOLO model.")
            return
        model.to(device)

        st.write(f"Running model {chosen_model} on {video_file.name}")
        summary = process_video(video_path, model)
        if summary is None:
            st.warning("No output returned.")

#########################
# TRANSFORMER TRAINING
#########################
def transformer_training_interface():
    st.title("Transformer Model Training (Frame+Class Multi-Label)")

    csv_file = st.file_uploader("Upload CSV with (frame,class_id)", type=["csv"])
    if csv_file is None:
        cfs = [f for f in os.listdir("runs") if f.endswith(".csv")]
        chosen = st.selectbox("Select CSV", cfs)
        if chosen:
            cpath = os.path.join("runs", chosen)
            st.session_state.csv_file_path = cpath
            st.success(f"Chosen => {chosen}")
        else:
            st.warning("No CSV chosen.")
    else:
        cpath = os.path.join("data", csv_file.name)
        with open(cpath,"wb") as f:
            f.write(csv_file.getbuffer())
        st.session_state.csv_file_path = cpath
        st.success(f"Saved => {csv_file.name}")

    if 'csv_file_path' in st.session_state:
        df_ = pd.read_csv(st.session_state.csv_file_path)
        st.write("CSV sample:")
        st.write(df_.head(10))

        if st.button("Train Transformer"):
            try:
                seq_len=20
                stride=10
                n_classes=7
                inputs, labels = prepare_transformer_inputs_from_csv(
                    df_, sequence_length=seq_len, num_classes=n_classes, stride=stride
                )
                st.write(f"Inputs => {inputs.shape}, Labels => {labels.shape}")
                model_ = train_transformer_model(
                    inputs, labels,
                    d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1,
                    num_classes=n_classes, num_epochs=100, batch_size=16
                )
                st.success("Transformer trained + saved.")
            except Exception as e:
                st.error(f"Error => {e}")

#########################
# TRANSFORMER MODEL LOADING
#########################
def load_transformer_model(
    model_path, d_model=64, nhead=2, num_layers=2, dim_feedforward=128,
    dropout=0.1, num_classes=7
):
    try:
        m_ = ActionRecognitionTransformer(
            input_size=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes
        ).to(device)
        st_ = torch.load(model_path, map_location=device)
        m_.load_state_dict(st_, strict=False)
        return m_
    except FileNotFoundError:
        st.error(f"Transformer model not found: {model_path}")
    except Exception as e:
        st.error(f"Error loading transformer: {e}")
    return None

#########################
# GPT Analysis
#########################
NUM_CLASSES = 7

########################################
# Simplified statistics function
########################################
def run_transformer_statistics(model_path, df_detections):
    """
    Simplified version:
      1) Loads model & runs inference to get presence of each class per frame.
      2) Counts how many frames each class is present in (no consecutive logic).
      3) Computes active vs rest frames based on presence of any punch/kick/guard classes.
      4) Displays bar chart and a simpler GPT summary.
    """
    # Load model
    model_ = load_transformer_model(model_path, num_classes=NUM_CLASSES)
    if model_ is None:
        st.error("Failed loading transformer model.")
        return

    # Convert CSV detections -> model input
    inputs, _ = prepare_transformer_inputs_from_csv(df_detections, sequence_length=20, stride=10)
    if inputs.shape[0] == 0:
        st.warning("No data found in CSV.")
        return

    # Predict
    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_logits = []
    with torch.no_grad():
        for i in range(len(inputs)):
            batch_in = inputs[i].unsqueeze(0).to(device_)
            logits = model_(batch_in)
            all_logits.append(logits.cpu())

    # Combine & threshold
    all_logits = torch.cat(all_logits, dim=0)  # shape => (N, seq_len, num_classes)
    probs = torch.sigmoid(all_logits)
    preds = (probs > 0.5).int().numpy()

    # Reconstruct presence per frame
    stride = 10
    seq_len = inputs.shape[1]
    global_presence = {}  # frame_idx -> [0..1 per class]
    N = preds.shape[0]    # number of chunks
    for chunk_idx in range(N):
        chunk_start = chunk_idx * stride
        for t in range(seq_len):
            f_idx = chunk_start + t
            if f_idx not in global_presence:
                global_presence[f_idx] = [0]*NUM_CLASSES
            for c_idx in range(NUM_CLASSES):
                global_presence[f_idx][c_idx] = max(
                    global_presence[f_idx][c_idx],
                    preds[chunk_idx, t, c_idx]
                )

    # Count frames per class
    frames_sorted = sorted(global_presence.keys())
    if not frames_sorted:
        st.warning("No frames in global predictions.")
        return
    total_frames = frames_sorted[-1] + 1

    frames_per_class = [0]*NUM_CLASSES
    for f_idx in frames_sorted:
        row = global_presence[f_idx]
        for c_idx in range(NUM_CLASSES):
            if row[c_idx] == 1:
                frames_per_class[c_idx] += 1

    # Active vs rest
    # Let's define "active" if cross, high-guard, hook, or kick is present
    # (class indices: 1=cross, 2=high-guard, 3=hook, 4=kick)
    active_frames = 0
    for f_idx in frames_sorted:
        row = global_presence[f_idx]
        if row[1] == 1 or row[2] == 1 or row[3] == 1 or row[4] == 1:
            active_frames += 1

    rest_frames = total_frames - active_frames
    active_ratio = 100.0 * active_frames / total_frames if total_frames > 0 else 0
    rest_ratio   = 100.0 * rest_frames   / total_frames if total_frames > 0 else 0

    # Show bar chart
    class_names = [CLASS_MAP[i] for i in range(NUM_CLASSES)]
    df_chart = pd.DataFrame({
        "Class": class_names,
        "Frames": frames_per_class
    }).set_index("Class")

    st.write("## Class Presence (Frames)")
    st.bar_chart(df_chart["Frames"])

    st.write("## Efficiency / Rest vs Active")
    st.write(f"Total Frames: {total_frames}")
    st.write(f"Active: {active_frames} ({active_ratio:.2f}%)")
    st.write(f"Rest:   {rest_frames}   ({rest_ratio:.2f}%)")

    # Simple textual breakdown
    frames_with_class = {
        CLASS_MAP[i]: frames_per_class[i] for i in range(NUM_CLASSES) if frames_per_class[i] > 0
    }
    text_analysis = generate_gpt_analysis(
        class_names=[k for k,v in frames_with_class.items()],
        frames_with_class=frames_with_class,
        total_frames=total_frames,
        active_ratio=active_ratio,
        rest_ratio=rest_ratio
    )

    st.write("## Simple GPT Insights")
    st.write(text_analysis)

    st.write("## CSV Preview")
    st.write(df_detections.head(10))

# ----------------------------------------------------------
#           GPT INSIGHTS
# ----------------------------------------------------------
def generate_gpt_analysis(
    class_names: List[str],
    frames_with_class: Dict[str, int],
    total_frames: int,
    active_ratio: float,
    rest_ratio: float
) -> str:
    """
    Produces a very simple English summary of which classes appeared,
    how many frames they appeared in, and a brief line on active/rest ratio.
    """
    if not class_names:
        return "No classes were detected in the session."

    lines = []
    lines.append(f"Detected classes: {', '.join(class_names)}.")
    lines.append(f"Active time: {active_ratio:.2f}% | Rest time: {rest_ratio:.2f}%.")

    # Show frames for each class
    for cname in class_names:
        frames_count = frames_with_class.get(cname, 0)
        lines.append(f"{cname} present in {frames_count} frames.")

    return " ".join(lines)

#########################
# RESULTS DASHBOARD
#########################
def transformer_results_dashboard():
    st.title("Results Dashboard (Transformer)")

    t_models = [f for f in os.listdir("models") if f.endswith(".pth")]
    chosen   = st.selectbox("Select Transformer Model", t_models)
    if chosen:
        model_path = os.path.join("models", chosen)
        st.success(f"Selected => {chosen}")

        # Let user pick any CSV in runs
        csv_files  = [f for f in os.listdir("runs") if f.endswith(".csv")]
        chosen_csv = st.selectbox("Select CSV File", csv_files)
        if chosen_csv:
            csv_path = os.path.join("runs", chosen_csv)
            st.success(f"Selected CSV => {chosen_csv}")
            if not os.path.exists(csv_path):
                st.error(f"File not found => {csv_path}")
            else:
                try:
                    df_ = pd.read_csv(csv_path)
                except Exception as e:
                    st.error(f"Reading CSV error => {e}")
                    return

                # Run stats
                run_transformer_statistics(model_path, df_)

#########################
# MAIN
#########################
def main():
    st.sidebar.title("Combat Sports Analysis - Transformer Time Series")
    app_mode = st.sidebar.selectbox(
       "Choose mode",
       ["Run Model","Transformer Training","Transformer Results"]
    )

    if app_mode == "Run Model":
        model_execution()
    elif app_mode == "Transformer Training":
        transformer_training_interface()
    elif app_mode == "Transformer Results":
        transformer_results_dashboard()

if __name__=="__main__":
    check_internet_connection()
    main()
