#########################
# FULL SCRIPT: TRANSFORMER FOR MULTI-LABEL TIME SERIES
#########################

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
pathlib.PosixPath = pathlib.WindowsPath

# Ensure essential dirs
for d in ['data','models','runs']:
    os.makedirs(d, exist_ok=True)

# Add local yolov5
yolov5_path = os.path.abspath("yolov5")
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

logging.basicConfig(level=logging.INFO)

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

# Uses GIT, will need to use program
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
        except subprocess.CalledProcessError as e:
            st.error("Failed to download YOLOv5. Please ensure Git is installed and accessible.")
            return False
    else:
        print("YOLOv5 repository already exists.")
    return True

#########################
# YAML CLASSES
#########################
# Load classes from YAML
def load_classes_from_yaml(yaml_path):
    try:
        # Force 'utf-8' with fallback
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

# Function to update paths in data.yaml
def update_yaml_paths(yaml_path, dataset_folder):
    if not yaml_path or not dataset_folder:
        raise ValueError("yaml_path and dataset_folder must be valid paths")

    try:
        # Verify the number of classes in the YAML file matches the model's expected number of classes
        with open(yaml_path, 'r') as file:  # Use yaml_path passed to the function
            yaml_data = yaml.safe_load(file)

        # Update the paths in the YAML file
        yaml_data['train'] = os.path.join(dataset_folder, 'train/images').replace('\\', '/')
        yaml_data['val'] = os.path.join(dataset_folder, 'valid/images').replace('\\', '/')
        yaml_data['test'] = os.path.join(dataset_folder, 'test/images').replace('\\', '/')

        # Write the updated YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(yaml_data, file)

        st.success(f"Successfully updated paths in {yaml_path}")

    except Exception as e:
        st.error(f"An error occurred while updating the YAML file: {e}")

#########################
# TIME-SERIES TRANSFORMER
#########################
class ActionRecognitionTransformer(nn.Module):
    """
    Sequence-to-sequence multi-label prediction using a Transformer encoder.
    We embed the multi-hot vectors, pass them through TransformerEncoder,
    then project back to multi-label logits for each time step.
    """
    def __init__(
        self,
        input_size=7,       # dimension of multi-hot
        d_model=64,         # embedding dimension for the Transformer
        nhead=2,            # number of heads
        num_layers=2,       # number of encoder layers
        dim_feedforward=128,# feedforward dimension
        dropout=0.1,
        num_classes=7       # output multi-label dimension
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
        # x shape: [batch_size, seq_len, input_size]
        emb = self.embedding(x) # => [batch_size, seq_len, d_model]
        out = self.transformer_encoder(emb) # => [batch_size, seq_len, d_model]
        logits = self.fc(out)  # => [batch_size, seq_len, num_classes]
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
    - We only read (frame, class_id) for multi-label logic.
    - For each frame, build a multi-hot vector of shape [num_classes].
    - Then chunk into sequences of length `sequence_length` with optional `stride`.
    - The label is the same shape as the input => we want to predict multi-hot for every time step.

    Returns: (all_inputs, all_labels)
      each => shape [N, sequence_length, num_classes]
    """
    if isinstance(csv_file_or_df, pd.DataFrame):
        df = csv_file_or_df
    else:
        df = pd.read_csv(csv_file_or_df)

    if 'frame' not in df.columns or 'class_id' not in df.columns:
        raise ValueError("CSV must have 'frame' and 'class_id' columns.")

    grouped = df.groupby('frame')
    frames = sorted(grouped.groups.keys())

    # Build a multi-hot for each frame
    # We'll store them in a list => frame_vectors[frame_idx] = [num_classes]
    max_frame = max(frames) if len(frames)>0 else 0
    frame_vectors = [[0]*num_classes for _ in range(max_frame+1)]
    for f_idx in frames:
        class_ids = grouped.get_group(f_idx)['class_id'].unique()
        for cid in class_ids:
            cid = int(cid)
            if cid>=0 and cid<num_classes:
                frame_vectors[f_idx][cid]=1

    # Now chunk them into sequences
    input_chunks = []
    label_chunks = []
    i=0
    while i+sequence_length<=len(frame_vectors):
        seq_data = frame_vectors[i:i+sequence_length]
        seq_tensor = torch.tensor(seq_data,dtype=torch.float)
        # For multi-label sequence-to-sequence, the label is the same shape
        input_chunks.append(seq_tensor)
        label_chunks.append(seq_tensor) # we want to reconstruct the same sequence
        i += stride

    if len(input_chunks)==0:
        return torch.empty(0), torch.empty(0)

    inputs = torch.stack(input_chunks) # [N, seq_len, num_classes]
    labels = torch.stack(label_chunks) # [N, seq_len, num_classes]
    return inputs, labels

#########################
# DATASET & LOADER
#########################
class TransformerDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels    = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

#########################
# TRAIN TRANSFORMER
#########################
def train_transformer_model(
    inputs, 
    labels,
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
    - BCEWithLogitsLoss over entire sequence:
      shape => [batch_size, seq_len, num_classes].
    - We'll flatten the last 2 dims or do per-step.
    """
    if isinstance(inputs, list):
        inputs = torch.stack(inputs)
    if isinstance(labels, list):
        labels = torch.stack(labels)

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("inputs shape:", inputs.shape, "labels shape:", labels.shape)

    N = len(inputs)
    if N<2:
        train_in=inputs
        train_lb=labels
        val_in=torch.empty(0)
        val_lb=torch.empty(0)
    else:
        split_idx = int(0.8*N)
        train_in  = inputs[:split_idx]
        train_lb  = labels[:split_idx]
        val_in    = inputs[split_idx:]
        val_lb    = labels[split_idx:]

    train_ds=TransformerDataset(train_in, train_lb)
    val_ds  =TransformerDataset(val_in, val_lb)

    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader  =DataLoader(val_ds,batch_size=batch_size,shuffle=False)

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
    
    scheduler=StepLR(optimizer, step_size=20,gamma=0.1)

    progress_bar = st.progress(0)
    epoch_text   = st.empty()
    best_val_loss=float('inf')
    patience=25
    patience_counter=0

    for epoch in range(num_epochs):
        # TRAIN
        model_.train()
        total_train_loss=0.0
        for seqb, lblb in train_loader:
            seqb=seqb.to(device_)   # [B, seq_len, num_classes]
            lblb=lblb.to(device_)   # same shape
            optimizer.zero_grad()
            outs=model_(seqb)       # => [B, seq_len, num_classes]
            # flatten => [B*seq_len, num_classes]
            loss=criterion(outs.view(-1,num_classes), lblb.view(-1,num_classes))
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()

        scheduler.step()

        # VAL
        val_loss=0.0
        model_.eval()
        if len(val_ds)>0:
            with torch.no_grad():
                for seqb,lblb in val_loader:
                    seqb=seqb.to(device_)
                    lblb=lblb.to(device_)
                    outs=model_(seqb)
                    l=criterion(outs.view(-1,num_classes), lblb.view(-1,num_classes))
                    val_loss+=l.item()

        epoch_text.text(f"Epoch[{epoch+1}/{num_epochs}] TrainLoss:{total_train_loss:.4f} ValLoss:{val_loss:.4f}")
        progress_bar.progress((epoch+1)/num_epochs)

        if len(val_ds)>0:
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                patience_counter=0
            else:
                patience_counter+=1
                if patience_counter>=patience:
                    st.warning("Early stopping triggered!")
                    try:
                        model_.load_state_dict(torch.load("models/transformer_model.pth"))
                    except:
                        pass
                    break

    final_path="models/transformer_model.pth"
    torch.save(model_.state_dict(), final_path)
    print(f"Transformer model saved at {final_path}")
    return model_

######################### 
# YOLO + Execution
#########################
### YOLO Functions
# Function to load YOLOv5 model
@st.cache_resource
def load_yolo_model(weights='models/best.pt'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Function to run YOLO inference on an image or frame
from utils.general import non_max_suppression
def yolo_inference(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    
    try:
        # Move the tensor to the CPU before converting to NumPy
        detections = results.xyxy[0].cpu().numpy()
    except AttributeError:
        st.error("YOLO model did not return a pandas DataFrame. Please check the model's output format.")
        st.write("Detect: ", detections)
        return []
    
    return detections

def calculate_iou(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    x_left = max(xA1, xB1)
    y_top = max(yA1, yB1)
    x_right = min(xA2, xB2)
    y_bottom = min(yA2, yB2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = (xA2 - xA1) * (yA2 - yA1)
    boxB_area = (xB2 - xB1) * (yB2 - yB1)
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    
    return iou

def boxes_intersect(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    return not (xA2 < xB1 or xB2 < xA1 or yA2 < yB1 or yB2 < yA1)

def process_video(video_path, model):
    class_names = st.session_state.get('yolo_classes', [])
    if not class_names:
        st.error("Class names are not available in session_state.")
        return []

    all_detections = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
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

    # Strike Counters
    counters = {
        'cross': 0,
        'hook': 0,
        'kick': 0
    }

    # Overlap Flags
    overlap_active = {
        'cross': False,
        'hook': False,
        'kick': False
    }

    # Store frame index of the last counted hit for each action
    # Place this OUTSIDE the while loop so it is NOT reinitialized every frame
    last_hit_frame = {
        'cross': -1,
        'hook': -1,
        'kick': -1
    }

    # You can set different grace periods per action if you wish:
    # for example, a cross might realistically be thrown again sooner than a kick.
    # The numbers below assume ~30 FPS and typical strike times.
    action_grace_period = {
        'cross': 6,   # ~0.2 seconds
        'hook': 8,    # ~0.27 seconds
        'kick': 15    # ~0.5 seconds
    }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    # Overlap Detection Function
    def check_overlap(action_boxes, boxing_bag_boxes, action_name, frame):
        """Return True if any action_box intersects any bag_box."""
        for action_box in action_boxes:
            for bag_box in boxing_bag_boxes:
                if boxes_intersect(action_box, bag_box):
                    # Optional debug output
                    centroid_action = ((action_box[0] + action_box[2]) // 2, (action_box[1] + action_box[3]) // 2)
                    centroid_bag = ((bag_box[0] + bag_box[2]) // 2, (bag_box[1] + bag_box[3]) // 2)
                    cv2.line(frame, centroid_action, centroid_bag, (0, 255, 255), 2)
                    return True
        return False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        detections = yolo_inference(frame, model)
        print(f"Frame {frame_count}: Detections: {detections}")

        # Initialize lists for each class
        class0_boxes = []  # Boxing Bag
        class1_boxes = []  # Cross
        class3_boxes = []  # Hook
        class4_boxes = []  # Kick
        class6_boxes = []  # Person

        CONFIDENCE_THRESHOLD = 0.7  # Adjust based on your model's performance

        for det in detections:
            # [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2 = map(int, [det[0], det[1], det[2], det[3]])
            confidence = det[4]
            class_id = int(det[5])

            if confidence < CONFIDENCE_THRESHOLD:
                continue  # Skip low-confidence detections

            if class_id == 0:
                color = (0, 255, 0)  # Green for boxing bag
                class0_boxes.append((x1, y1, x2, y2))
            elif class_id == 1:
                color = (255, 0, 255)  # Magenta for cross
                class1_boxes.append((x1, y1, x2, y2))
            elif class_id == 3:
                color = (255, 165, 0)  # Orange for hook
                class3_boxes.append((x1, y1, x2, y2))
            elif class_id == 4:
                color = (0, 255, 255)  # Yellow for kick
                class4_boxes.append((x1, y1, x2, y2))
            elif class_id == 6:
                color = (255, 0, 0)  # Blue for person
                class6_boxes.append((x1, y1, x2, y2))
            else:
                color = (0, 0, 255)  # Red for other classes

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[class_id]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate and draw centroid
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, centroid, 5, color, 2)
            cv2.putText(frame, str(centroid), centroid, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Debugging statement
        print(f"Frame {frame_count}: Detected Classes - Cross: {len(class1_boxes)}, "
              f"Hook: {len(class3_boxes)}, Kick: {len(class4_boxes)}, "
              f"Boxing Bags: {len(class6_boxes)}")

        # Check overlaps for each action
        # Use class0_boxes (the true boxing-bag) for overlap
        overlap_results = {
            'cross': check_overlap(class1_boxes, class0_boxes, 'Cross', frame),
            'hook':  check_overlap(class3_boxes, class0_boxes, 'Hook', frame),
            'kick':  check_overlap(class4_boxes, class0_boxes, 'Kick', frame)
        }

        # Update counters based on overlaps + grace periods
        for action in ['cross', 'hook', 'kick']:
            if overlap_results[action]:
                # Overlap is true this frame
                if not overlap_active[action]:
                    # Overlap just STARTED this frame
                    current_frame = frame_count
                    # Check if enough frames have passed since last counted
                    if (current_frame - last_hit_frame[action]) > action_grace_period[action]:
                        counters[action] += 1
                        last_hit_frame[action] = current_frame
                        print(f"Frame {frame_count}: Counted a {action} hit. Total so far: {counters[action]}")
                    overlap_active[action] = True
            else:
                # No overlap this frame
                overlap_active[action] = False

        # Calculate total hits
        hit_detected = counters['cross'] + counters['hook'] + counters['kick']

        # Display the hit counters on the frame
        cv2.putText(frame, f"Hit: {hit_detected}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Cross: {counters['cross']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Hook: {counters['hook']}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        cv2.putText(frame, f"Kick: {counters['kick']}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Optionally, collect all detections
        if isinstance(detections, np.ndarray) and detections.size > 0:
            all_detections.append(detections)

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    # Release video resources
    cap.release()
    out.release()
    progress_bar.empty()

    st.success(
        f"Processing complete.\n"
        f"Total Cross hits: {counters['cross']}\n"
        f"Total Hook hits: {counters['hook']}\n"
        f"Total Kick hits: {counters['kick']}"
    )
    return all_detections

#########################
#   LOAD TRANSFORMER MODEL
#########################
def load_transformer_model(model_path, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1, num_classes=7):
    try:
        model_=ActionRecognitionTransformer(
            input_size=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes
        ).to(device)
        state_dict=torch.load(model_path,map_location=device)
        model_.load_state_dict(state_dict, strict=False)
        return model_
    except FileNotFoundError:
        st.error(f"Transformer model not found: {model_path}")
    except Exception as e:
        st.error(f"Error loading: {e}")
    return None

#########################
#   APP SECTIONS
#########################
#########################
#   DATA COLLECTION
#########################
def data_collection():
    st.title("Data Collection")
    st.write("Upload videos and images to build your dataset. Use external tools to label your data.")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['mp4', 'avi', 'mov', 'jpg', 'png'])

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            valid_extensions = ['.mp4', '.avi', '.mov', '.jpg', '.png']
            if file_extension in valid_extensions:
                sanitized_filename = os.path.basename(uploaded_file.name)
                save_path = os.path.join(data_dir, sanitized_filename)
                
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"File '{sanitized_filename}' uploaded successfully!")
                except Exception as e:
                    st.error(f"Failed to save '{sanitized_filename}': {e}")
            else:
                st.warning(f"File '{uploaded_file.name}' is not a valid file type and was not saved.")
    
    st.subheader("Labeling Tools")
    st.write("Use external tools like LabelImg or CVAT to label your data.")
    st.markdown("- [Roboflow](https://app.roboflow.com/)")
    st.markdown("- [LabelImg GitHub](https://github.com/tzutalin/labelImg)")
    st.markdown("- [CVAT GitHub](https://github.com/opencv/cvat)")

#########################
#   YOLO TRAIN UI
#########################
def yolo_training():
    st.title("YOLO Model Training")
    st.subheader("Select Dataset Location and Update YAML")

    # 1) Check for a "default data.yaml" in dataset/
    default_yaml_path = os.path.join(os.getcwd(), "dataset", "data.yaml")
    
    # 2) Let user optionally upload a new data.yaml
    st.write("If you don't upload a file, we'll try to use dataset/data.yaml by default.")
    user_file = st.file_uploader("Upload data.yaml", type="yaml")

    # 3) Decide which YAML to use: user upload or default
    if user_file is not None:
        # -- USER FILE CASE --
        # We'll treat the user file like your original code
        dataset_folder = os.path.join(os.getcwd(), "dataset")
        os.makedirs(dataset_folder, exist_ok=True)

        yaml_file_path = os.path.join(dataset_folder, "data.yaml")
        with open(yaml_file_path, "wb") as file:
            file.write(user_file.getbuffer())

        st.info(f"Uploaded file saved as: {yaml_file_path}")
    else:
        # -- DEFAULT FILE CASE --
        if os.path.exists(default_yaml_path):
            st.info(f"No file uploaded, using default data.yaml: {default_yaml_path}")
            dataset_folder = os.path.join(os.getcwd(), "dataset")
            yaml_file_path = default_yaml_path
        else:
            st.warning("No data.yaml uploaded and no default found at dataset/data.yaml. Cannot proceed.")
            return  # Stop here, because we have no valid YAML

    # 4) Try to update paths in the chosen YAML
    try:
        update_yaml_paths(yaml_file_path, dataset_folder)
        st.session_state['dataset_folder'] = dataset_folder
        st.session_state['data_yaml_path'] = yaml_file_path

        # 5) Load classes
        yolo_classes = load_classes_from_yaml(yaml_file_path)
        if yolo_classes:
            st.success(f"Classes loaded: {yolo_classes}")
        else:
            st.error("Error loading classes from YAML.")
    except Exception as e:
        st.error(f"An error occurred while updating the YAML file: {str(e)}")
        return

    # 6) Now we confirm that dataset_folder and data_yaml_path exist
    dataset_path = st.session_state.get('dataset_folder')
    data_yaml_path = st.session_state.get('data_yaml_path')

    if dataset_path and data_yaml_path and os.path.exists(data_yaml_path):
        # 7) Standard YOLO training params
        st.subheader("Training Parameters")
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=100)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
        img_size = st.number_input("Image Size (pixels)", min_value=320, max_value=1920, value=640, step=32)
        yolo_model = st.selectbox("Select YOLO Model", ["yolov5s", "yolov5m", "yolov5l", "yolov5x"])

        if st.button("Start YOLO Training"):
            if not os.path.exists(dataset_path):
                st.error(f"Dataset path '{dataset_path}' not found.")
                return
            if not os.path.exists(data_yaml_path):
                st.error(f"YAML file '{data_yaml_path}' not found.")
                return

            st.write("Starting YOLO model training...")

            # 8) Path to YOLOv5 train.py
            yolo_path = os.path.abspath("yolov5")
            train_py_path = os.path.join(yolo_path, "train.py")
            if not os.path.exists(train_py_path):
                st.error(f"train.py not found at {train_py_path}. Please check the YOLOv5 installation.")
                return

            command = [
                sys.executable, train_py_path,
                '--img', str(img_size),
                '--batch', str(batch_size),
                '--epochs', str(epochs),
                '--data', data_yaml_path,
                '--weights', f'{yolo_model}.pt',
                '--device', '0'  # or 'cpu' if no GPU
            ]

            # 9) Launch training & show logs
            progress_bar = st.progress(0)
            output_box = st.empty()
            try:
                with st.spinner('Training in progress...'):
                    process = subprocess.Popen(
                        command,
                        cwd=yolo_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=1,
                        universal_newlines=True
                    )

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
                    st.success("YOLO model trained successfully!")
                    progress_bar.progress(100)
                else:
                    st.error("YOLO model training failed.")
                    st.write(process.stderr.read())
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("No valid dataset or data.yaml. Please provide a valid data.yaml.")

#########################
#  MODEL EXECUTION
#########################
# Main Run Function
def model_execution():
    st.title("Model Execution")

    # Step 0: Select data.yaml file (Prompt at the beginning)
    default_yaml_path = os.path.join("dataset", "data.yaml")
    yaml_file_path = None

    if os.path.exists(default_yaml_path):
        st.info(f"Default data.yaml file found at {default_yaml_path}")
        yaml_file_path = default_yaml_path

    uploaded_yaml_file = st.file_uploader("Select data.yaml file", type=["yaml"])
    if uploaded_yaml_file is not None:
        yaml_file_path = os.path.join("dataset", uploaded_yaml_file.name)
        with open(yaml_file_path, "wb") as f:
            f.write(uploaded_yaml_file.getbuffer())
        st.success(f"Uploaded YAML file: {uploaded_yaml_file.name}")

    if yaml_file_path is None:
        st.warning("Please upload the data.yaml file before proceeding.")
        return

    # st.success(f"Using data.yaml file: {yaml_file_path}")
    yolo_classes = load_classes_from_yaml(yaml_file_path)

    # Step 1: Select YOLO Model
    yolo_models = [f for f in os.listdir('models') if f.endswith('.pt')]
    selected_yolo_model = st.selectbox("Select YOLO Model", yolo_models)

    model_path = os.path.join('models', selected_yolo_model)

    # Step 2: Upload video
    video_file = st.file_uploader("Upload video file for action recognition", type=['mp4', 'avi', 'mov'])
    if video_file:
        video_path = os.path.join("data", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        # st.success(f"Uploaded video: {video_file.name}")

    # Step 3: Only run the models after the button is clicked
    if st.button("Run Models") and video_file:
        # Load the YOLO model
        model = load_yolo_model(model_path)
        model = model.to(device)

        if model is None:
            st.error("Failed to load the YOLO model.")
            return

        # st.success(f"Running model '{selected_yolo_model}' on video '{video_file.name}'")
        detections = process_video(video_path, model)

        if not detections:
            st.error("No detections were made on the video.")
            return

        # Ensure that we check for valid detections before processing
        detected_class_ids = []
        detected_class_names = []
        all_detections = []  # This will store detections for each frame

        # Loop through each frame and add the frame index to the detections
        for frame_idx, frame in enumerate(detections):  # Now tracking frame index
            frame_detections = []
            for d in frame:
                if len(d) >= 6:  # Ensure the detection has at least 6 elements
                    class_id = int(d[5])  # Access class ID at index 5
                    if 0 <= class_id < len(yolo_classes):
                        detected_class_ids.append(class_id)
                        # Append frame index along with the detection info
                        frame_detections.append([frame_idx, *d[:6]])  # Add frame index as the first element
                    else:
                        st.warning(f"Class ID {class_id} exceeds the number of available YOLO classes.")
            all_detections.append(frame_detections)

        # Save the predictions into a CSV file for the results dashboard, with a unique name
        csv_file_path = os.path.join("runs", f"yolo_predictions_{video_file.name}.csv")
        # Flatten all_detections (list of lists) and save it as a DataFrame
        df_detections = pd.DataFrame([det for frame in all_detections for det in frame], 
                                     columns=['frame', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id'])
        df_detections.to_csv(csv_file_path, index=False)
        st.success(f"Predictions saved to {csv_file_path}.")

# Helper function to check if two bounding boxes overlap
def check_overlap(box1, box2):
    """Helper function to check if two bounding boxes overlap."""
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    return not (x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1)

#########################
# TRAIN UI (TRANSFORMER)
#########################
def transformer_training_interface():
    st.title("Transformer Model Training (Frame+Class Multi-Label)")

    csv_file=st.file_uploader("Upload CSV with (frame,class_id)", type=['csv'])
    if csv_file is None:
        cfs=[f for f in os.listdir('runs') if f.endswith('.csv')]
        chosen=st.selectbox("Select CSV", cfs)
        if chosen:
            path_=os.path.join("runs", chosen)
            st.session_state.csv_file_path=path_
            st.success(f"Selected CSV: {chosen}")
        else:
            st.warning("No CSV chosen.")
    else:
        path_=os.path.join("data", csv_file.name)
        with open(path_,"wb") as fil:
            fil.write(csv_file.getbuffer())
        st.session_state.csv_file_path=path_
        st.success(f"CSV uploaded: {csv_file.name}")

    if 'csv_file_path' in st.session_state:
        df_=pd.read_csv(st.session_state.csv_file_path)
        st.write("Sample of CSV:")
        st.write(df_.head(20))

        if st.button("Train Transformer"):
            # Prepare chunked sequences
            try:
                seq_len=20
                stride=10
                num_classes=7
                inputs, labels = prepare_transformer_inputs_from_csv(
                    df_, sequence_length=seq_len, num_classes=num_classes, stride=stride
                )
                st.write(f"Prepared shape: inputs={inputs.shape}, labels={labels.shape}")

                model_=train_transformer_model(
                    inputs, labels,
                    d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1,
                    num_classes=num_classes,
                    num_epochs=100, batch_size=16
                )
                st.success("Transformer model trained & saved.")
            except Exception as e:
                st.error(f"Error training: {e}")

#########################
# RUN STATISTICS / RESULTS
#########################

def generate_gpt_analysis(actions_sequence: List[str], stats_summary: Dict) -> str:
    """
    Generates detailed textual insights based on actions sequence and statistics summary.
    """
    # Descriptive Introduction
    if not actions_sequence:
        # In case there's no detected action at all
        description = "The recorded session had no significant detections. "
    elif len(actions_sequence) == 1:
        # Only one type of action
        description = f"The recorded session encompasses a variety of detections including {actions_sequence[0]}. "
    else:
        description = (
            f"The recorded session encompasses a variety of detections including {', '.join(actions_sequence[:-1])}, "
            f"and {actions_sequence[-1]}. "
        )

    # Extract statistics
    instances = stats_summary.get("instances", [])
    durations = stats_summary.get("durations", [])
    rest_ratio = stats_summary.get("rest_ratio", 0.0)
    active_ratio = stats_summary.get("active_ratio", 0.0)
    training_frames = stats_summary.get("training_frames", 0)
    sparring_frames = stats_summary.get("sparring_frames", 0)
    total_frames = stats_summary.get("total_frames", 1)  # Avoid division by zero
    class_map = stats_summary.get("class_map", {})

    # Detailed Statistics Extraction
    # Safely get the indexes for cross/hook/kick
    idx_cross = class_map.get("cross", 1)
    idx_hook  = class_map.get("hook", 3)
    idx_kick  = class_map.get("kick", 4)

    cross_count = instances[idx_cross] if idx_cross < len(instances) else 0
    hook_count  = instances[idx_hook]  if idx_hook  < len(instances) else 0
    kick_count  = instances[idx_kick]  if idx_kick  < len(instances) else 0
    punch_count = cross_count + hook_count
    total_actions = punch_count + kick_count

    # Generate Insights
    insights = []

    # Efficiency Metrics
    insights.append(
        f"During the session, the athlete was actively engaged in movements for {active_ratio:.2f}% of the time, "
        f"while allocating {rest_ratio:.2f}% for rest and recovery periods."
    )

    # Action Frequency Analysis
    if total_actions > 0:
        insights.append(f"A total of {cross_count} cross punches were performed.")
        insights.append(f"A total of {hook_count} hook punches were performed.")
        insights.append(f"A total of {punch_count} punches were performed.")
        insights.append(f"A total of {kick_count} kicks were performed.")

    # Action Type Dominance
    if total_actions > 0:
        if punch_count > kick_count:
            punch_ratio = (punch_count / total_actions) * 100
            insights.append(
                f"Punches constitute {punch_ratio:.2f}% of the total actions, highlighting a strong emphasis on boxing techniques."
            )
        elif kick_count > punch_count:
            kick_ratio = (kick_count / total_actions) * 100
            insights.append(
                f"Kicks make up {kick_ratio:.2f}% of the total actions, suggesting a focus on kickboxing or martial arts training."
            )
        else:
            insights.append(
                "The athlete maintains a balanced approach between punching and kicking techniques, "
                "ensuring versatility in combat skills."
            )

    # Activity Context
    person_idx = class_map.get("person", 6)
    bag_idx    = class_map.get("boxing-bag", 0)
    if person_idx < len(durations) and bag_idx < len(durations):
        person_presence = (durations[person_idx] / total_frames) * 100
        bag_presence    = (durations[bag_idx]    / total_frames) * 100
        if person_presence > 90 and bag_presence > 90:
            insights.append("The session predominantly features an individual training intensely with a boxing bag.")
        elif sparring_frames > training_frames:
            insights.append("The session mainly involves sparring activities between two participants, showcasing interactive combat practice.")
        elif training_frames > sparring_frames:
            insights.append("The session is focused on individual training routines, emphasizing technique and endurance drills.")
        else:
            insights.append("The session includes a balanced mix of training and sparring activities without a clear dominance of either.")
    else:
        insights.append("Unable to determine training context due to missing person/bag data.")

    # Compile final text
    full_insight = description + " ".join(insights)
    return full_insight


def run_transformer_statistics(model_path, df_detections, video_path=None):
    """
    Enhanced multi-label stats & GPT analysis for Transformer-based predictions,
    with flicker filtering and correct frame counting.
    """

    ###################################################
    # 1) Load Transformer model
    ###################################################
    model_ = load_transformer_model(
        model_path,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=7
    )
    if model_ is None:
        st.error("Failed to load transformer.")
        return

    ###################################################
    # 2) Prepare data as chunked inputs
    ###################################################
    seq_len = 20
    stride = 10
    num_classes = 7
    inputs, _ = prepare_transformer_inputs_from_csv(
        df_detections,
        sequence_length=seq_len,
        num_classes=num_classes,
        stride=stride
    )
    if inputs.shape[0] == 0:
        st.warning("No sequences found.")
        return

    ###################################################
    # 3) Inference
    ###################################################
    model_.eval()
    with torch.no_grad():
        outs_list = []
        for i in range(len(inputs)):
            seq_in = inputs[i].unsqueeze(0).to(device)
            logits = model_(seq_in)  # shape: [1, seq_len, num_classes]
            outs_list.append(logits.cpu())
        all_logits = torch.cat(outs_list, dim=0)  # shape [N, seq_len, num_classes]

    probs = torch.sigmoid(all_logits)
    preds = (probs > 0.5).int().numpy()  # shape [N, seq_len, num_classes]

    ###################################################
    # 4) Merge chunk predictions => global_preds
    ###################################################
    global_preds = {}
    N = len(inputs)
    for i in range(N):
        chunk_start = i * stride
        for t_ in range(seq_len):
            frame_idx = chunk_start + t_
            if frame_idx not in global_preds:
                global_preds[frame_idx] = [0]*num_classes
            for c_idx in range(num_classes):
                # "OR" the presence over overlapping chunks
                global_preds[frame_idx][c_idx] = max(global_preds[frame_idx][c_idx], preds[i, t_, c_idx])

    frames_sorted = sorted(global_preds.keys())
    if not frames_sorted:
        st.warning("No frames in global predictions.")
        return

    # Actual total frames (not sum of durations!)
    real_total_frames = frames_sorted[-1] + 1

    ###################################################
    # 5) Flicker filtering + instance counting
    ###################################################
    class_map = {
        0: "boxing-bag",
        1: "cross",
        2: "high-guard",
        3: "hook",
        4: "kick",
        5: "low-guard",
        6: "person"
    }

    # For each class, we require X consecutive frames to count an instance
    MIN_CONSECUTIVE = {
        0: 1,  # bag
        1: 2,  # cross
        2: 2,  # high-guard
        3: 2,  # hook
        4: 2,  # kick
        5: 1,  # low-guard
        6: 1   # person
    }

    instance_counts = [0]*num_classes
    durations = [0]*num_classes

    # We'll track how long each class has been consecutively active
    consecutive_streak = [0]*num_classes
    # We also track whether the class is in the "middle" of a run we've counted
    instance_in_run = [False]*num_classes

    # Training vs Sparring counters
    training_frames = 0
    sparring_frames = 0

    for f_idx in frames_sorted:
        current = global_preds[f_idx]

        # Evaluate training vs sparring
        # (simple rule: if person(6) + bag(0) => training; if 2 persons => sparring, etc.)
        num_person = current[6]  # 0 or 1
        num_bag    = current[0]  # 0 or 1
        if num_person > 1:
            # This would be a multi-person scenario (not typical with 0/1 predictions).
            sparring_frames += 1
        elif num_person == 1 and num_bag == 1:
            training_frames += 1

        # For each class, update streaks + durations
        for c_idx in range(num_classes):
            if current[c_idx] == 1:
                durations[c_idx] += 1
                consecutive_streak[c_idx] += 1
            else:
                consecutive_streak[c_idx] = 0
                instance_in_run[c_idx] = False

            # Check if we can count a new instance
            if consecutive_streak[c_idx] >= MIN_CONSECUTIVE[c_idx] and not instance_in_run[c_idx]:
                instance_counts[c_idx] += 1
                instance_in_run[c_idx] = True

    ###################################################
    # 6) Compute rest vs active ratios using real frames
    ###################################################
    # We'll define "active" as any presence of cross(1), high-guard(2), hook(3), or kick(4) in a frame.
    # We'll define "resting" otherwise (e.g., bag, low-guard, person).
    active_frame_count = 0
    for f_idx in frames_sorted:
        current = global_preds[f_idx]
        # If any of cross/hook/kick/high-guard is 1 => active
        if current[1] == 1 or current[2] == 1 or current[3] == 1 or current[4] == 1:
            active_frame_count += 1

    rest_frame_count = real_total_frames - active_frame_count
    active_ratio = (active_frame_count / real_total_frames) * 100 if real_total_frames > 0 else 0
    rest_ratio   = (rest_frame_count   / real_total_frames) * 100 if real_total_frames > 0 else 0

    ###################################################
    # 7) Visualization + GPT summary
    ###################################################
    # Show bar chart of durations
    st.write("## Class Durations (Frames)")
    chart_data = pd.DataFrame({
        'Class': [class_map[i] for i in range(num_classes)],
        'DurationFrames': durations
    })
    chart_data.set_index('Class', inplace=True)
    st.bar_chart(chart_data['DurationFrames'])

    st.write("## Athlete Efficiency / Rest vs Active")
    st.write(f"Real Total Frames: {real_total_frames}")
    st.write(f"Active Frames: {active_frame_count} ({active_ratio:.2f}%)")
    st.write(f"Resting Frames: {rest_frame_count} ({rest_ratio:.2f}%)")

    st.write("## Class Instance Counts (After Flicker Filtering)")
    for i in range(num_classes):
        st.write(f"{class_map[i]}: {instance_counts[i]}")

    # Prepare data for GPT analysis
    # Build a reverse map: "boxing-bag" -> 0, "cross" -> 1, etc.
    reverse_map = {v: k for k, v in class_map.items()}
    stats_summary = {
        "instances": instance_counts,
        "durations": durations,
        "rest_ratio": rest_ratio,
        "active_ratio": active_ratio,
        "training_frames": training_frames,
        "sparring_frames": sparring_frames,
        "total_frames": real_total_frames,
        "class_map": {
            "boxing-bag": reverse_map.get("boxing-bag", 0),
            "cross":      reverse_map.get("cross", 1),
            "high-guard": reverse_map.get("high-guard", 2),
            "hook":       reverse_map.get("hook", 3),
            "kick":       reverse_map.get("kick", 4),
            "low-guard":  reverse_map.get("low-guard", 5),
            "person":     reverse_map.get("person", 6)
        }
    }

    # Build a list of actions that actually appeared
    actions_sequence = []
    for i in range(num_classes):
        if instance_counts[i] > 0:
            actions_sequence.append(class_map[i])

    # Generate GPT analysis
    gpt_result = generate_gpt_analysis(actions_sequence, stats_summary)

    st.write("## GPT Insights")
    st.write(gpt_result)

    st.write("## CSV Preview")
    st.write(df_detections.head(5))

#########################
#  RESULTS DASHBOARD
#########################
def transformer_results_dashboard():
    st.title("Results Dashboard (Transformer)")

    # Select Transformer model
    t_models = [f for f in os.listdir('models') if f.endswith('.pth')]
    selected_model = st.selectbox("Select Transformer Model", t_models)
    if selected_model:
        model_path = os.path.join("models", selected_model)
        st.success(f"Selected model: {selected_model}")

        # Select CSV
        csv_files = [f for f in os.listdir('runs') if f.endswith('.csv')]
        selected_csv = st.selectbox("Select CSV File", csv_files)
        if selected_csv:
            csv_path = os.path.join("runs", selected_csv)
            st.success(f"Selected CSV: {selected_csv}")

            # Check if the CSV file exists
            if not os.path.exists(csv_path):
                st.error(f"CSV file '{selected_csv}' not found in the 'runs' directory.")
            else:
                # Read the CSV file
                try:
                    df_ = pd.read_csv(csv_path)
                    # st.write("### CSV Content Preview:")
                    # st.dataframe(df_.head())  # Display the first few rows of the CSV
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")

                # Define the prefix and suffix to remove
                prefix = "yolo_predictions_"
                suffix = ".csv"

                # Verify that the CSV filename starts with the prefix and ends with the suffix
                if selected_csv.startswith(prefix) and selected_csv.endswith(suffix):
                    # Extract the video filename by removing the prefix and suffix
                    video_filename = selected_csv[len(prefix):-len(suffix)]
                    # st.write(f"Derived Video Filename: **{video_filename}**")
                else:
                    st.error("CSV filename does not follow the expected format 'yolo_predictions_<video_name>.csv'.")
                    video_filename = None  # Set to None to avoid further processing

                # Proceed only if video_filename was successfully extracted
                if video_filename:
                    # Construct the video path relative to the current working directory
                    video_path = os.path.join("runs", video_filename)

                    # Display the video path for debugging (optional)
                    # st.write(f"Video Path: {video_path}")

                    # Check if the video file exists
                    if os.path.exists(video_path):
                        st.success(f"Video file '{video_filename}' found.")

                        # Add a button to open the video in the default media player
                        if st.button("Open Video in Default Media Player"):
                            try:
                                os.startfile(video_path)  # Opens with the default media player on Windows
                                st.info("Opening video...")
                            except Exception as e:
                                st.error(f"Failed to open video: {e}")
                    else:
                        st.error(f"Video file '{video_filename}' not found in the 'runs' directory.")

            # Run stats function
            run_transformer_statistics(model_path, df_)

#########################
# MAIN APP
#########################
def main():
    st.sidebar.title("Combat Sports Analysis - Transformer Time Series")
    app_mode=st.sidebar.selectbox(
        "Choose mode",
        ["Run Model","Data Collection","YOLO Training","Transformer Training","Transformer Results"]
    )
    if app_mode=="Run Model":
        model_execution()
    elif app_mode=="Data Collection":
        data_collection()
    elif app_mode=="YOLO Training":
        yolo_training()
    elif app_mode=="Transformer Training":
        transformer_training_interface()
    elif app_mode=="Transformer Results":
        transformer_results_dashboard()

if __name__=="__main__":
    check_internet_connection()
    main()
