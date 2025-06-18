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

def prepare_transformer_inputs_from_csv(csv_file_or_df, sequence_length=None, stride=None, num_classes=None):
    # 0) defaulting + validation
    DEFAULT_SEQ_LEN = 14
    if sequence_length is None:
        sequence_length = DEFAULT_SEQ_LEN
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive int, got {sequence_length!r}")

    if stride is None:
        stride = sequence_length // 2
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
                seq_len = 14
                stride  = seq_len // 2
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
      - Computes dynamic grace periods from data for hit counting
      - Builds a multi-frame actions_sequence for Markov analysis
      - Compresses consecutive repeats before HMM analysis
      - Reports detailed statistics and distribution metrics
    """
    import matplotlib.pyplot as plt

    # 1) Load the Transformer model
    model_ = load_transformer_model(
        model_path,
        d_model=64, nhead=2,
        num_layers=2, dim_feedforward=128,
        dropout=0.1, num_classes=6
    )
    if model_ is None:
        st.error("No transformer model found.")
        return

    # 2) Prepare the transformer inputs
    DEFAULT_SEQ_LEN = 14
    seq_len = DEFAULT_SEQ_LEN
    stride = seq_len // 2
    num_classes = 6
    inputs, _ = prepare_transformer_inputs_from_csv(
        df_detections,
        sequence_length=seq_len,
        stride=stride,
        num_classes=num_classes
    )
    if inputs.shape[0] == 0:
        st.warning("No valid input sequences.")
        return

    # 3) Model inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_.to(device).eval()
    with torch.no_grad():
        logits = torch.cat([
            model_(inputs[i].unsqueeze(0).to(device)).cpu()
            for i in range(inputs.shape[0])
        ], dim=0)

    # 4) Threshold to binary predictions
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)  # threshold

    # 5) Aggregate sliding-window outputs into per-frame labels AND confidences
    global_preds = {}
    global_probs = {}   # ← define your confidence store

    for win_idx in range(preds.shape[0]):
        start = win_idx * stride
        for t in range(seq_len):
            frame    = start + t
            arr_pred = preds[win_idx, t]
            arr_prob = probs[win_idx, t]

            if frame not in global_preds:
                global_preds[frame] = np.zeros(num_classes, dtype=int)
                global_probs[frame] = np.zeros(num_classes, dtype=float)  # ← init here

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
    # keep track of which frames contain hits
    punch_frames = set()
    kick_frames  = set()

    # 7) Build frame-by-frame action sequence
    sequence = []
    # ← INSERT THIS: keep track of which frames contain hits
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

    # ← INSERT THIS BLOCK immediately after the loop, 
    #     before you use df_hits in your Cumulative Hits plot:
    df_hits = pd.DataFrame({'frame': frames})
    df_hits['punches'] = df_hits['frame'].isin(punch_frames).cumsum()
    df_hits['kicks']   = df_hits['frame'].isin(kick_frames).cumsum()

    df_seq = pd.DataFrame({'frame': frames, 'action': sequence})

    # 8) Segment lengths for each action type
    df_seq = pd.DataFrame({'frame': frames, 'action': sequence})
    df_seq['segment'] = (df_seq['action'] != df_seq['action'].shift()).cumsum()
    segs = df_seq.groupby('segment').agg(
        action=('action', 'first'),
        start=('frame', 'first'),
        end=('frame', 'last')
    ).reset_index(drop=True)
    segs['length'] = segs['end'] - segs['start'] + 1

    # === NEW: Count hits by contiguous segments instead of frames ===
    # Create a DataFrame covering ALL frames in the video
    df_hits = pd.DataFrame({'frame': range(total_frames)})
    
    # Identify start frames for punch/kick segments
    punch_starts = segs[segs['action'] == 'punch']['start'].tolist()
    kick_starts = segs[segs['action'] == 'kick-knee']['start'].tolist()
    
    # Mark event frames and compute cumulative counts
    df_hits['punch_event'] = df_hits['frame'].isin(punch_starts).astype(int)
    df_hits['kick_event'] = df_hits['frame'].isin(kick_starts).astype(int)
    df_hits['punches'] = df_hits['punch_event'].cumsum()
    df_hits['kicks'] = df_hits['kick_event'].cumsum()

    # 11) Compute durations and ratios
    durations = df_seq['action'].value_counts().reindex(class_map.values(), fill_value=0)
    active = df_seq[df_seq['action'].isin(['punch','kick-knee'])].shape[0]
    active_ratio = active / total_frames * 100

    # 12) Display results
    st.write(f"Total frames: {total_frames}")
    st.write(f"Punch segments: {len(segs[segs['action']=='punch'])},  Kick segments: {len(segs[segs['action']=='kick-knee'])}")
    st.write(f"Active vs rest frames: {active} vs {total_frames-active} ({active_ratio:.1f}% active)")

    # 13) Show distributions
    st.write("### Segment length distributions (frame counts)")
    st.write(segs.groupby('action')['length'].describe())

    chart_data = pd.DataFrame({
        'Action': durations.index,
        'DurationFrames': durations.values
    }).set_index('Action')
    st.bar_chart(chart_data)

    # 10) Stats over segments
    stats = segs.groupby('action')['length'].describe().rename(columns={
        'count':'Segments', 'mean':'AvgLen', 'min':'MinLen', '25%':'Q1','50%':'Median','75%':'Q3','max':'MaxLen'
    })

    # 12) Compressed sequence and transitions
    compressed = compress_state_sequence(sequence)

    # Transition counts
    trans = pd.DataFrame({'prev': compressed[:-1], 'next': compressed[1:]})
    trans_mat = pd.crosstab(trans['prev'], trans['next']).reindex(index=stats.index, columns=stats.index, fill_value=0)
    st.write("### Transition matrix (count of transitions)")
    st.dataframe(trans_mat)

    # 14) HMM analysis on compressed sequence
    compressed = compress_state_sequence(sequence)
    st.write(build_and_run_hmm(compressed))

    # 14) Show sample detections
    st.write("### CSV Preview")
    st.write(df_detections.head(5))

    # --- Action Timeline (Gantt) ---
    from matplotlib.patches import Patch
    st.write("### Action Timeline")
    fig_timeline, ax = plt.subplots(figsize=(8, 2))
    color_map = {'punch': 'red', 'kick-knee': 'blue', 'high-guard': 'green', 'idle': 'gray'}

    # draw each segment
    for _, row in segs.iterrows():
        ax.barh(0,
                row['length'],
                left=row['start'],
                color=color_map[row['action']],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5)

    # add grid for frame ticks
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # legend
    handles = [Patch(color=col, label=act) for act, col in color_map.items()]
    ax.legend(handles=handles,
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.2),
            loc='upper center',
            frameon=False)

    # tidy up axes
    ax.set_yticks([])
    ax.set_xlabel('Frame')

    st.pyplot(fig_timeline, use_container_width=False)


    st.write("### Segment-Length Histogram")
    # --- Segment-Length Histogram ---
    fig_hist, ax = plt.subplots()
    for act, grp in segs.groupby('action'):
        ax.hist(grp['length'], bins=10, alpha=0.5, label=act)
    ax.set_xlabel('Segment Length (frames)'); ax.set_ylabel('Count'); ax.legend()
    st.pyplot(fig_hist, use_container_width=False)

    st.write("### Cumulative Hits Over Time")
    # --- Cumulative Hits Over Time ---
    df_hits = pd.DataFrame({'frame': frames})
    df_hits['punches'] = df_hits['frame'].isin(punch_frames).cumsum()
    df_hits['kicks']   = df_hits['frame'].isin(kick_frames).cumsum()
    fig_cum_hits, ax = plt.subplots()
    ax.plot(df_hits['frame'], df_hits['punches'], label='Punches')
    ax.plot(df_hits['frame'], df_hits['kicks'],   label='Kicks')
    ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Hits'); ax.legend()
    st.pyplot(fig_cum_hits, use_container_width=False)

    st.write("### Inter-Hit Interval Distribution")
    # --- Inter-Hit Interval Distribution ---
    intervals_p = np.diff(sorted(punch_frames)) if len(punch_frames)>1 else []
    intervals_k = np.diff(sorted(kick_frames))  if len(kick_frames)>1  else []
    fig_intervals, ax = plt.subplots()
    if len(intervals_p): ax.hist(intervals_p, bins=10, alpha=0.5, label='Punch')
    if len(intervals_k): ax.hist(intervals_k, bins=10, alpha=0.5, label='Kick')
    ax.set_xlabel('Frames Between Hits'); ax.set_ylabel('Count'); ax.legend()
    st.pyplot(fig_intervals, use_container_width=False)

    st.write("### Transition Probability Heatmap")
    # --- Transition Probability Heatmap ---
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

    st.write("### Max Class Confidence Over Time")
    # --- Max Class Confidence Over Time ---
    df_conf = pd.DataFrame({
        'frame': frames,
        'max_conf': [global_probs[f].max() for f in frames]
    })
    fig_conf, ax = plt.subplots()
    ax.plot(df_conf['frame'], df_conf['max_conf'])
    ax.hlines(0.8, xmin=0, xmax=total_frames, linestyles='dashed')
    ax.set_xlabel('Frame'); ax.set_ylabel('Max Confidence')
    st.pyplot(fig_conf, use_container_width=False)

    st.write("### Active-Classes Frequency Distribution")
    # --- Active-Classes Frequency Distribution ---
    df_seq['active_count'] = df_seq['frame'].apply(lambda f: global_preds[f].sum())
    counts = df_seq['active_count'].value_counts().sort_index()

    fig_active_dist, ax = plt.subplots(figsize=(4,2))
    ax.bar(counts.index, counts.values, width=0.6, edgecolor='black')
    ax.set_xlabel('Active Classes per Frame')
    ax.set_ylabel('Number of Frames')
    ax.set_xticks(counts.index)
    st.pyplot(fig_active_dist, use_container_width=False)

    st.write("### State Transition Network")
    # --- State Transition Network ---
    import networkx as nx  # make sure this is at the top of your file

    G = nx.DiGraph()
    for (u, v), w in prob_mat.stack().items():
        if w > 0:
            G.add_edge(u, v, weight=w)

    fig_network, ax = plt.subplots(figsize=(7, 7))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, ax=ax)

    # build edge_labels separately to avoid inline f-string issues
    edge_labels = { (u, v): f"{d['weight']:.2f}" 
                    for (u, v), d in G.edges.items() }

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax
    )
    st.pyplot(fig_network, use_container_width=False)

    # Download All Results
    import io, zipfile

    # 1) Gather your figures into a dict (use the variables you used when plotting)
    charts = {
        "action_timeline.png":         fig_timeline,
        "segment_histogram.png":       fig_hist,
        "cumulative_hits.png":         fig_cum_hits,
        "inter_hit_intervals.png":     fig_intervals,
        "transition_heatmap.png":      fig_heatmap,
        "max_confidence.png":          fig_conf,
        "active_classes_dist.png":     fig_active_dist,
        "state_transition_network.png":fig_network
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