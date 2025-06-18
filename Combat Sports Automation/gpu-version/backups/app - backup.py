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

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    GPT2LMHeadModel = None
    GPT2Tokenizer = None

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("torch").setLevel(logging.ERROR)
st.set_page_config(page_title="Combat Sports Prototype", layout="wide")
pathlib.PosixPath = pathlib.WindowsPath

if "dataset_folder" not in st.session_state:
    st.session_state["dataset_folder"] = None
if "data_yaml_path" not in st.session_state:
    st.session_state["data_yaml_path"] = None
if "yolo_classes" not in st.session_state:
    st.session_state["yolo_classes"] = []

def check_internet_connection():
    try:
        socket.create_connection(("www.google.com",80), timeout=5)
        return True
    except OSError:
        return False

def download_yolov5():
    if not os.path.exists('yolov5'):
        git_exists = shutil.which("git")
        if git_exists is None:
            st.error("Git not installed.")
            return False
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'], check=True)
            st.success("Downloaded YOLOv5.")
        except subprocess.CalledProcessError:
            st.error("Failed to clone YOLOv5.")
            return False
    return True

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

def prepare_transformer_inputs_from_csv(csv_file_or_df, sequence_length=20, stride=10, num_classes=None):
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

class TransformerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

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

def load_local_gpt2(model_name="gpt2"):
    if "gpt2_model" not in st.session_state or st.session_state["gpt2_model"] is None:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        st.session_state["gpt2_model"] = model
        st.session_state["gpt2_tokenizer"] = tokenizer
    return st.session_state["gpt2_model"], st.session_state["gpt2_tokenizer"]

def generate_text_with_gpt2(prompt, max_new_tokens=50, temperature=0.7):
    if GPT2LMHeadModel is None or GPT2Tokenizer is None:
        return "GPT not available."
    model, tokenizer = load_local_gpt2()
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@st.cache_resource
def load_yolo_model(weights='models/best.pt'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def yolo_inference(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    try:
        detections = results.xyxy[0].cpu().numpy()
    except:
        return []
    return detections

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

def process_video(video_path, model):
    class_names = st.session_state.get('yolo_classes', [])
    if not class_names:
        st.error("No class names.")
        return []
    all_detections = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening {video_path}")
        return []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = "runs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_count = 0
    counters = {'punch': 0, 'kick-knee': 0}
    overlap_active = {'punch': False, 'kick-knee': False}
    last_hit_frame = {'punch': -1, 'kick-knee': -1}
    action_grace_period = {'punch': 2, 'kick-knee': 15}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    def check_overlap_(action_boxes, bag_boxes, action_name, frame_):
        for a_box in action_boxes:
            for b_box in bag_boxes:
                if boxes_intersect(a_box, b_box):
                    centroid_action = ((a_box[0] + a_box[2]) // 2, (a_box[1] + a_box[3]) // 2)
                    centroid_bag = ((b_box[0] + b_box[2]) // 2, (b_box[1] + b_box[3]) // 2)
                    cv2.line(frame_, centroid_action, centroid_bag, (0, 255, 255), 2)
                    return True
        return False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = yolo_inference(frame, model)
        class0_boxes = []
        class2_boxes = []
        class5_boxes = []
        class4_boxes = []
        class1_boxes = []
        class3_boxes = []
        CONFIDENCE_THRESHOLD = 0.7
        for det in detections:
            x1, y1, x2, y2 = map(int, [det[0], det[1], det[2], det[3]])
            confidence = det[4]
            class_id = int(det[5])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            if class_id == 0:
                color = (0, 255, 0)
                class0_boxes.append((x1, y1, x2, y2))
            elif class_id == 2:
                color = (0, 255, 255)
                class2_boxes.append((x1, y1, x2, y2))
            elif class_id == 5:
                color = (255, 0, 255)
                class5_boxes.append((x1, y1, x2, y2))
            elif class_id == 4:
                color = (255, 0, 0)
                class4_boxes.append((x1, y1, x2, y2))
            elif class_id == 1:
                color = (0, 128, 255)
                class1_boxes.append((x1, y1, x2, y2))
            elif class_id == 3:
                color = (255, 165, 0)
                class3_boxes.append((x1, y1, x2, y2))
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[class_id]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, centroid, 5, color, 2)
            cv2.putText(frame, str(centroid), centroid, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
        overlap_results = {}
        overlap_results['punch'] = check_overlap_(class5_boxes, class0_boxes, 'punch', frame)
        overlap_results['kick-knee'] = check_overlap_(class2_boxes, class0_boxes, 'kick-knee', frame)
        for action in ['punch', 'kick-knee']:
            if overlap_results[action]:
                if not overlap_active[action]:
                    current_frame = frame_count
                    if (current_frame - last_hit_frame[action]) > action_grace_period[action]:
                        counters[action] += 1
                        last_hit_frame[action] = current_frame
                    overlap_active[action] = True
            else:
                overlap_active[action] = False
        total_hits = counters['punch'] + counters['kick-knee']
        cv2.putText(frame, f"Hits: {total_hits}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Punch: {counters['punch']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Kick-Knee: {counters['kick-knee']}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out.write(frame)
        if isinstance(detections, np.ndarray) and detections.size > 0:
            all_detections.append(detections)
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))
    cap.release()
    out.release()
    progress_bar.empty()
    st.success(f"Done.\nPunch: {counters['punch']}\nKick-Knee: {counters['kick-knee']}")
    return all_detections

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

def data_collection():
    st.title("Data Collection")
    st.write("Upload videos/images for dataset.")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['mp4','avi','mov','jpg','png'])
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
    st.subheader("Labeling Tools")
    st.write("Use external tools to label your data.")
    st.markdown("- Roboflow\n- LabelImg\n- CVAT")

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

def model_execution():
    st.title("Model Execution")
    default_yaml_path = os.path.join("dataset", "data.yaml")
    yaml_file_path = None
    if os.path.exists(default_yaml_path):
        yaml_file_path = default_yaml_path
    uploaded_yaml_file = st.file_uploader("Select data.yaml file", type=["yaml"])
    if uploaded_yaml_file is not None:
        yaml_file_path = os.path.join("dataset", uploaded_yaml_file.name)
        with open(yaml_file_path, "wb") as f:
            f.write(uploaded_yaml_file.getbuffer())
        st.success(f"Uploaded YAML: {uploaded_yaml_file.name}")
    if yaml_file_path is None:
        st.warning("Please upload data.yaml first.")
        return
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
        csv_file_path = os.path.join("runs", f"yolo_predictions_{video_file.name}.csv")
        df_detections = pd.DataFrame([det for frame in all_dets for det in frame], columns=['frame','x1','y1','x2','y2','confidence','class_id'])
        df_detections.to_csv(csv_file_path, index=False)
        st.success(f"Predictions => {csv_file_path}")

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
                seq_len = 20
                stride = 10
                num_classes = 6
                inputs, labels = prepare_transformer_inputs_from_csv(df_, sequence_length=seq_len, num_classes=num_classes, stride=stride)
                st.write(f"inputs={inputs.shape}, labels={labels.shape}")
                model_ = train_transformer_model(inputs, labels, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=num_classes, num_epochs=100, batch_size=16)
                st.success("Transformer trained.")
            except Exception as e:
                st.error(f"Error training: {e}")

def generate_gpt_analysis(stats_summary):
    instances = stats_summary.get("instances", [])
    durations = stats_summary.get("durations", [])
    rest_ratio = stats_summary.get("rest_ratio", 0.0)
    active_ratio = stats_summary.get("active_ratio", 0.0)
    training_frames = stats_summary.get("training_frames", 0)
    sparring_frames = stats_summary.get("sparring_frames", 0)
    total_frames = stats_summary.get("total_frames", 1)
    class_map = stats_summary.get("class_map", {})
    idx_punch = class_map.get("punch", 5)
    idx_kickknee = class_map.get("kick-knee", 2)
    punch_count = instances[idx_punch] if idx_punch < len(instances) else 0
    kick_count = instances[idx_kickknee] if idx_kickknee < len(instances) else 0
    total_actions = punch_count + kick_count
    insights = []
    insights.append(f"Active {active_ratio:.2f}% of the time, resting {rest_ratio:.2f}%.")
    if total_actions > 0:
        insights.append(f"{punch_count} punch.")
        insights.append(f"{kick_count} kick-knee.")
    if total_actions > 0:
        if punch_count > kick_count:
            punch_ratio = (punch_count / total_actions)*100
            insights.append(f"Punch => {punch_ratio:.2f}% of total.")
        elif kick_count > punch_count:
            kick_ratio = (kick_count / total_actions)*100
            insights.append(f"Kick-Knee => {kick_ratio:.2f}% of total.")
        else:
            insights.append("Balanced approach.")
    bag_idx = class_map.get("boxing-bag", 0)
    person_idx = class_map.get("person", 4)
    if bag_idx < len(durations) and person_idx < len(durations):
        person_presence = (durations[person_idx]/total_frames)*100
        bag_presence = (durations[bag_idx]/total_frames)*100
        if person_presence>90 and bag_presence>90:
            insights.append("Single athlete with bag.")
        elif sparring_frames>training_frames:
            insights.append("More sparring than solo training.")
        elif training_frames>sparring_frames:
            insights.append("More solo training than sparring.")
        else:
            insights.append("Mixed training.")
    else:
        insights.append("Missing context for person/bag.")
    full_insight = insights
    return full_insight

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


def run_transformer_statistics(model_path, df_detections, video_path=None):
    """
    Transformer-based approach that:
      - Infers punch/kick/high-guard presence from frame-level predictions
      - Applies a grace period for final "hit" counts (punch, kick)
      - Builds a multi-frame actions_sequence for Markov analysis
      - Compresses consecutive repeats (idle->idle->idle => idle) 
        before calling build_and_run_hmm
    """
    # 1) Load the Transformer model
    model_ = load_transformer_model(
        model_path,
        d_model=64, nhead=2,
        num_layers=2, dim_feedforward=128,
        dropout=0.1, num_classes=6
    )
    if model_ is None:
        st.error("No transformer.")
        return

    # 2) Prepare the data
    seq_len = 20
    stride  = 10
    num_classes = 6
    inputs, _ = prepare_transformer_inputs_from_csv(
        df_detections,
        sequence_length=seq_len,
        num_classes=num_classes,
        stride=stride
    )
    if inputs.shape[0] == 0:
        st.warning("No sequences.")
        return

    # 3) Forward pass
    model_.eval()
    with torch.no_grad():
        outs_list = []
        for i in range(len(inputs)):
            seq_in = inputs[i].unsqueeze(0).to(device)
            logits = model_(seq_in)
            outs_list.append(logits.cpu())
        all_logits = torch.cat(outs_list, dim=0)

    # 4) Threshold predictions
    probs = torch.sigmoid(all_logits)
    preds = (probs > 0.8).int().numpy()  # adjust threshold as needed

    # 5) Build per-frame predictions
    global_preds = {}
    N = len(inputs)
    for i in range(N):
        chunk_start = i * stride
        for t_ in range(seq_len):
            frame_idx = chunk_start + t_
            if frame_idx not in global_preds:
                global_preds[frame_idx] = [0]*num_classes
            for c_idx in range(num_classes):
                global_preds[frame_idx][c_idx] = max(
                    global_preds[frame_idx][c_idx],
                    preds[i, t_, c_idx]
                )

    frames_sorted = sorted(global_preds.keys())
    if not frames_sorted:
        st.warning("No frames in global predictions.")
        return

    real_total_frames = frames_sorted[-1] + 1

    # 6) Class map: 0=bag,1=high-guard,2=kick-knee,3=low-guard,4=person,5=punch
    # We'll interpret index=1 => "high-guard"
    class_map = {
        0: "boxing-bag",
        1: "high-guard",
        2: "kick-knee",
        3: "low-guard",
        4: "person",
        5: "punch"
    }

    # 7) Count durations for each class
    durations = [0]*num_classes
    for f_idx in frames_sorted:
        for c_idx in range(num_classes):
            if global_preds[f_idx][c_idx] == 1:
                durations[c_idx] += 1

    # 8) YOLO-like final hits with grace period
    punch_grace   = 8
    kick_grace    = 10
    last_punch_fr = -9999
    last_kick_fr  = -9999
    punch_count   = 0
    kick_count    = 0

    punch_frames = set()
    kick_frames  = set()

    # 9) Build the raw frame-by-frame sequence
    actions_sequence = []
    for f_idx in frames_sorted:
        current = global_preds[f_idx]

        bag_present      = (current[0] == 1)  # class=0 => bag
        high_guard_state = (current[1] == 1)  # class=1 => high-guard
        kick_state       = (current[2] == 1)  # class=2 => kick-knee
        punch_state      = (current[5] == 1)  # class=5 => punch

        # For final numeric counting
        if punch_state:
            punch_frames.add(f_idx)
        if kick_state:
            kick_frames.add(f_idx)

        # Count hits with grace
        if punch_state and bag_present:
            if (f_idx - last_punch_fr) > punch_grace:
                punch_count += 1
                last_punch_fr = f_idx

        if kick_state and bag_present:
            if (f_idx - last_kick_fr) > kick_grace:
                kick_count += 1
                last_kick_fr = f_idx

        # Build the action label each frame
        if punch_state and bag_present:
            actions_sequence.append("punch")
        elif kick_state and bag_present:
            actions_sequence.append("kick-knee")
        elif high_guard_state:
            actions_sequence.append("high-guard")
        else:
            actions_sequence.append("idle")

    total_hits = punch_count + kick_count

    # 10) Active vs rest
    active_frames = punch_frames.union(kick_frames)
    active_frame_count = len(active_frames)
    active_ratio = (active_frame_count / real_total_frames)*100 if real_total_frames>0 else 0.0
    rest_ratio  = 100.0 - active_ratio

    # 11) Display numeric results
    st.write(f"Punches (Transformer logic): {punch_count}")
    st.write(f"Kicks   (Transformer logic): {kick_count}")
    st.write(f"Total Hits: {total_hits}")
    st.write(f"Active Ratio: {active_ratio:.2f}%, Rest Ratio: {rest_ratio:.2f}%")

    # Show bar chart of durations
    chart_data = pd.DataFrame({
        'Class': [class_map[i] for i in range(num_classes)],
        'DurationFrames': durations
    }).set_index('Class')
    st.bar_chart(chart_data['DurationFrames'])

    # 12) Stats summary for GPT
    stats_summary = {
        "instances": [
            punch_count if i == 5 else
            kick_count  if i == 2 else 0
            for i in range(num_classes)
        ],
        "durations": durations,
        "rest_ratio": rest_ratio,
        "active_ratio": active_ratio,
        "training_frames": 0,
        "sparring_frames": 0,
        "total_frames": real_total_frames,
        "class_map": {
            "boxing-bag": 0,
            "high-guard": 1,
            "kick-knee": 2,
            "low-guard": 3,
            "person": 4,
            "punch": 5
        }
    }

    # 13) GPT summary
    gpt_result = generate_gpt_analysis(stats_summary)
    st.write("GPT Insights:")
    st.write(gpt_result)

    # 14) Compress consecutive states
    compressed_sequence = compress_state_sequence(actions_sequence)

    # 15) Markov analysis on compressed states
    final_markov = build_and_run_hmm(compressed_sequence)
    st.write("Markov Model Insights:")
    st.write(final_markov)

    # 16) Show CSV snippet
    st.write("CSV Preview:")
    st.write(df_detections.head(5))

def transformer_results_dashboard():
    st.title("Results Dashboard (Transformer)")
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
    if st.button("Generate Stats & GPT Summary"):
        csv_path = os.path.join("runs", sel_csv)
        df_ = pd.read_csv(csv_path)
        run_transformer_statistics(os.path.join("models", sel_t), df_, sel_csv)

def main():
    st.sidebar.title("Combat Sports Analysis - Transformer Time Series")
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