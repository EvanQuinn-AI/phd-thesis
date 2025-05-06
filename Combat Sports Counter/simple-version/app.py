#########################
# Improved Customer-Facing App with 30 FPS Normalization
# and Advanced Strike Counting, No Console Outputs
#########################

import os
import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import yaml
import time
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure essential dirs
for d in ['data','models','runs']:
    os.makedirs(d, exist_ok=True)

import sys
import subprocess
import shutil

# Attempt YOLOv5 download if not present (no console prints, just Streamlit messages)
if not os.path.exists('yolov5'):
    git_exists = shutil.which("git")
    if git_exists:
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'], check=True)
            st.success("YOLOv5 repository downloaded successfully.")
        except subprocess.CalledProcessError:
            st.error("Failed to download YOLOv5. Please ensure Git is installed and accessible.")
    else:
        st.warning("Git not found; skipping YOLOv5 clone attempt.")

yolov5_path = os.path.abspath("yolov5")
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

#########################
# SESSION STATE
#########################
if "webcam_running" not in st.session_state:
    st.session_state["webcam_running"] = False

#########################
# Enumerate Video Devices
#########################
def enumerate_video_devices(max_tests=10):
    """
    Attempt to open camera indices from 0..max_tests-1.
    Return a list of device indices that are available.
    """
    available = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

#########################
# Convert any uploaded video to 30 FPS
#########################
def convert_video_to_30fps(input_path, output_path):
    """
    Reads all frames from 'input_path' and writes them
    to 'output_path' at exactly 30 FPS, using the same resolution.
    This changes playback speed if the source FPS != 30.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"Could not open file for 30FPS conversion: {input_path}")
        return False

    target_fps = 30.0  
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return True

#########################
# LOAD CLASSES
#########################
def load_classes(yaml_path="dataset/data.yaml"):
    """
    Loads class names from a YOLO-style data.yaml.
    Expects a 'names' key listing classes in order.
    """
    if not os.path.exists(yaml_path):
        st.error(f"data.yaml not found at: {yaml_path}. Please ensure it exists.")
        return []
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            names = data.get('names', [])
            return names
    except Exception as e:
        st.error(f"Failed to load YAML {yaml_path}: {e}")
        return []

#########################
# YOLO MODEL LOADING
#########################
@st.cache_resource
def load_yolo_model(weights='models/best.pt'):
    """
    Loads YOLOv5 model from local weights.
    If you face caching or path issues on Windows, set force_reload=True.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

#########################
# BOX-INTERSECTION & STRIKE COUNTING
#########################
def boxes_intersect(boxA, boxB):
    """Check if two bounding boxes (x1,y1,x2,y2) intersect."""
    return not (boxA[2] < boxB[0] or boxB[2] < boxA[0] or boxA[3] < boxB[1] or boxB[3] < boxA[1])

def detect_hits_on_frame(
    frame,
    detections,
    class_names,
    counters,
    overlap_active,
    last_hit_frame,
    action_grace_period,
    current_frame
):
    """
    For each detection, if we see cross/hook/kick intersect with bag, we increment
    counters only if:
      - We haven't been in 'overlap' for that action
      - Enough frames have passed since last counted
    """
    bag_boxes   = []
    cross_boxes = []
    hook_boxes  = []
    kick_boxes  = []

    # For bounding box color-coding
    color_map = {
        0: (0,255,0),     # bag => green
        1: (255,0,255),   # cross => magenta
        3: (255,165,0),   # hook => orange
        4: (0,255,255),   # kick => yellow
    }
    CONFIDENCE_THRESHOLD = 0.7

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)  # ensure int
        if conf < CONFIDENCE_THRESHOLD:
            continue

        if 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = f"cls_{cls_id}"

        color = color_map.get(cls_id, (0,0,255))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (int(x1), int(y1)-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        # Sort bounding boxes by class
        if cls_id == 0:
            bag_boxes.append((x1,y1,x2,y2))
        elif cls_id == 1:
            cross_boxes.append((x1,y1,x2,y2))
        elif cls_id == 3:
            hook_boxes.append((x1,y1,x2,y2))
        elif cls_id == 4:
            kick_boxes.append((x1,y1,x2,y2))

    # Overlap check
    def any_overlap(action_boxes, bag_boxes):
        for abox in action_boxes:
            for bbox in bag_boxes:
                if boxes_intersect(abox, bbox):
                    return True
        return False

    cross_ov = any_overlap(cross_boxes, bag_boxes)
    if cross_ov:
        if not overlap_active['cross']:
            if (current_frame - last_hit_frame['cross']) > action_grace_period['cross']:
                counters['cross'] += 1
                last_hit_frame['cross'] = current_frame
        overlap_active['cross'] = True
    else:
        overlap_active['cross'] = False

    hook_ov = any_overlap(hook_boxes, bag_boxes)
    if hook_ov:
        if not overlap_active['hook']:
            if (current_frame - last_hit_frame['hook']) > action_grace_period['hook']:
                counters['hook'] += 1
                last_hit_frame['hook'] = current_frame
        overlap_active['hook'] = True
    else:
        overlap_active['hook'] = False

    kick_ov = any_overlap(kick_boxes, bag_boxes)
    if kick_ov:
        if not overlap_active['kick']:
            if (current_frame - last_hit_frame['kick']) > action_grace_period['kick']:
                counters['kick'] += 1
                last_hit_frame['kick'] = current_frame
        overlap_active['kick'] = True
    else:
        overlap_active['kick'] = False

    total_hits = counters['cross'] + counters['hook'] + counters['kick']
    cv2.putText(frame, f"Hit: {total_hits}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Cross: {counters['cross']}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    cv2.putText(frame, f"Hook: {counters['hook']}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0), 2)
    cv2.putText(frame, f"Kick: {counters['kick']}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)


#########################
# MAIN INFERENCE LOGIC
#########################
def run_live_inference():
    """
    Allows user to either:
      - Upload Video
      - Use Webcam
    Runs YOLO with advanced strike counting. Saves annotated video & CSV.
    Videos standardized to 30 FPS to ensure consistent processing.
    """
    st.title("Run Live Model Inference (30FPS, Advanced Strike Counting)")

    class_names = load_classes("dataset/data.yaml")
    if not class_names:
        return

    action_grace_period = {
        'cross': 6,   # 0.2s at 30FPS
        'hook': 8,    # ~0.27s
        'kick': 15    # ~0.5s
    }
    counters = {'cross': 0, 'hook': 0, 'kick': 0}
    overlap_active = {'cross': False, 'hook': False, 'kick': False}
    last_hit_frame = {'cross': -1, 'hook': -1, 'kick': -1}

    model = load_yolo_model('models/best.pt')
    if model is None:
        return

    input_choice = st.radio("Select Input Source", ["Upload Video", "Webcam"])

    if input_choice == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4","avi","mov"])
        if uploaded_video:
            # Save local original
            original_path = os.path.join("data", uploaded_video.name)
            with open(original_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            st.success(f"Uploaded video: {uploaded_video.name}")

            # Convert to 30FPS
            base_name = os.path.splitext(uploaded_video.name)[0]
            standard_30_path = os.path.join("data", f"{base_name}_30fps.mp4")
            st.info("Converting video to 30 FPS (this may take a moment)...")
            ok = convert_video_to_30fps(original_path, standard_30_path)
            if not ok:
                st.error("Failed to convert video to 30 FPS.")
                return
            st.success("Conversion to 30 FPS complete.")

            if st.button("Start Detection"):
                cap = cv2.VideoCapture(standard_30_path)
                stframe = st.empty()

                out_vid_name = f"processed_{int(time.time())}.mp4"
                out_path = os.path.join("runs", out_vid_name)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                all_detections = []
                frame_id = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model(rgb)
                    try:
                        detections = results.xyxy[0].cpu().numpy()
                    except:
                        detections = np.array([])

                    detect_hits_on_frame(
                        frame, detections, class_names,
                        counters, overlap_active, last_hit_frame,
                        action_grace_period, current_frame=frame_id
                    )

                    for det in detections:
                        x1,y1,x2,y2,conf,cls_id = det
                        cls_id = int(cls_id)
                        all_detections.append([frame_id, x1,y1,x2,y2, conf, cls_id])

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels='RGB', use_column_width=True)

                    out_writer.write(frame)
                    frame_id += 1

                cap.release()
                out_writer.release()
                st.success("Finished processing the 30FPS video.")
                st.video(out_path)

                csv_name = f"yolo_predictions_{base_name}_30fps.csv"
                csv_path = os.path.join("runs", csv_name)
                df = pd.DataFrame(all_detections, columns=["frame","x1","y1","x2","y2","confidence","class_id"])
                df.to_csv(csv_path, index=False)
                st.success(f"Saved CSV: {csv_path}")

                total_hits = counters['cross'] + counters['hook'] + counters['kick']
                st.info(f"Cross: {counters['cross']}, Hook: {counters['hook']}, Kick: {counters['kick']}, Total: {total_hits}")

    elif input_choice == "Webcam":
        st.warning("Ensure you have a working webcam. Some browsers may not support it.")

        devices = enumerate_video_devices(max_tests=10)
        if not devices:
            st.error("No webcam devices found.")
            return
        device_strs = [f"Camera {d}" for d in devices]
        device_choice = st.selectbox("Select a Webcam Device", device_strs)
        cam_index = devices[device_strs.index(device_choice)]

        start_btn = st.button("Start Webcam Detection")
        stop_btn  = st.button("Stop Webcam Detection")

        if start_btn:
            # Initialize counters, overlap flags, etc. here if you want them reset each time
            counters = {'cross': 0, 'hook': 0, 'kick': 0}
            overlap_active = {'cross': False, 'hook': False, 'kick': False}
            last_hit_frame = {'cross': -1, 'hook': -1, 'kick': -1}

            st.session_state["webcam_running"] = True

            # Prepare CSV path and open it immediately
            csv_name_webcam = f"yolo_predictions_webcam_{int(time.time())}.csv"
            csv_path_webcam = os.path.join("runs", csv_name_webcam)

            import csv
            csv_file = open(csv_path_webcam, mode="w", newline="")
            csv_writer = csv.writer(csv_file)
            # Write CSV header
            csv_writer.writerow(["frame","x1","y1","x2","y2","confidence","class_id"])

            # Also prepare video writer
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name=='nt' else 0)
            stframe = st.empty()

            out_vid_name = f"webcam_{int(time.time())}.mp4"
            out_path = os.path.join("runs", out_vid_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0  # forcing 30 FPS output
            width, height = 640, 480
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            frame_id = 0

            while st.session_state["webcam_running"]:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb)
                try:
                    detections = results.xyxy[0].cpu().numpy()
                except:
                    detections = np.array([])

                detect_hits_on_frame(
                    frame, detections, class_names,
                    counters, overlap_active, last_hit_frame,
                    action_grace_period, current_frame=frame_id
                )

                # Write each detection row to CSV immediately
                for det in detections:
                    x1,y1,x2,y2,conf,cls_id = det
                    cls_id = int(cls_id)
                    csv_writer.writerow([frame_id, x1, y1, x2, y2, conf, cls_id])

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels='RGB')

                out_writer.write(frame)
                frame_id += 1

            # Once the loop ends, close everything
            cap.release()
            out_writer.release()
            csv_file.close()  # <--- This finalizes your CSV
            st.success(f"Stopped webcam. Video saved to {out_path}")
            st.success(f"CSV saved to {csv_path_webcam}")

            total_hits = counters['cross'] + counters['hook'] + counters['kick']
            st.info(f"Cross: {counters['cross']}, Hook: {counters['hook']}, Kick: {counters['kick']}, Total: {total_hits}")

        if stop_btn:
            # This button simply triggers the loop to end
            st.session_state["webcam_running"] = False


#########################
# TRANSFORMER MODEL
#########################
class ActionRecognitionTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=7):
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

def load_transformer_model(model_path, num_classes):
    try:
        model_ = ActionRecognitionTransformer(input_size=num_classes, num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=device)
        model_.load_state_dict(state_dict)
        model_.to(device)
        model_.eval()
        return model_
    except Exception as e:
        st.error(f"Error loading transformer model: {e}")
        return None

def prepare_inputs_for_transformer(csv_file, num_classes, seq_len=20, stride=10):
    df = pd.read_csv(csv_file)
    if "frame" not in df.columns or "class_id" not in df.columns:
        raise ValueError("CSV must contain 'frame' and 'class_id'.")

    grouped = df.groupby('frame')
    frames = sorted(grouped.groups.keys())
    if not frames:
        return torch.empty(0), []

    max_frame = max(frames)
    frame_vectors = [[0]*num_classes for _ in range(max_frame+1)]

    for f_idx in frames:
        class_ids = grouped.get_group(f_idx)['class_id'].unique()
        for cid in class_ids:
            cid = int(cid)
            if 0 <= cid < num_classes:
                frame_vectors[f_idx][cid] = 1

    input_chunks = []
    i=0
    while i+seq_len <= len(frame_vectors):
        seq_data = frame_vectors[i:i+seq_len]
        seq_tensor = torch.tensor(seq_data, dtype=torch.float)
        input_chunks.append(seq_tensor)
        i += stride

    if len(input_chunks)==0:
        return torch.empty(0), []

    return torch.stack(input_chunks), frames

def generate_gpt_insights(text):
    return f"GPT Analysis: {text}"

def transformer_analysis_dashboard():
    st.title("Transformer Analysis Dashboard")

    class_names = load_classes("dataset/data.yaml")
    num_classes = len(class_names)
    if num_classes == 0:
        return

    transformer_path = "models/transformer_model.pth"
    model_ = load_transformer_model(transformer_path, num_classes)
    if model_ is None:
        return

    csv_files = [f for f in os.listdir("runs") if f.endswith(".csv")]
    if not csv_files:
        st.warning("No CSV files found in 'runs'. Please run a detection first.")
        return

    selected_csv = st.selectbox("Select YOLO Detections CSV", csv_files)
    if selected_csv:
        csv_path = os.path.join("runs", selected_csv)
        st.info(f"Selected CSV: {csv_path}")

        if st.button("Run Transformer Analysis"):
            try:
                inputs, frames = prepare_inputs_for_transformer(csv_path, num_classes)
                if inputs.shape[0] == 0:
                    st.warning("No sequences found. Possibly too few frames or empty CSV.")
                    return

                model_.eval()
                all_preds = []
                with torch.no_grad():
                    for i in range(len(inputs)):
                        seq_in = inputs[i].unsqueeze(0).to(device)
                        logits = model_(seq_in)
                        all_preds.append(logits.cpu())

                all_logits = torch.cat(all_preds, dim=0)
                all_probs  = torch.sigmoid(all_logits)
                all_bin = (all_probs>0.5).float()
                class_counts = all_bin.sum(dim=(0,1)).numpy().astype(int)

                st.write("## Class Occurrences (Raw Counting)")
                chart_data = pd.DataFrame({
                    'Classes': class_names,
                    'Count': class_counts
                })
                st.bar_chart(chart_data.set_index('Classes'))

                summary_text = " | ".join(f"{class_names[i]}: {class_counts[i]}" for i in range(num_classes))
                st.info(summary_text)

                gpt_result = generate_gpt_insights(summary_text)
                st.write("## GPT Analysis")
                st.write(gpt_result)
            except Exception as e:
                st.error(f"Error running Transformer Analysis: {e}")

#########################
# MAIN
#########################
def main():
    st.sidebar.title("Customer-Facing Analysis")
    app_mode = st.sidebar.selectbox(
        "Select a Mode",
        ["Run Model", "Transformer Analysis"]
    )

    if app_mode == "Run Model":
        run_live_inference()
    elif app_mode == "Transformer Analysis":
        transformer_analysis_dashboard()

if __name__ == "__main__":
    main()
