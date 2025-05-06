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
from torch.utils.data import Dataset, DataLoader

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
    from torch.optim.lr_scheduler import StepLR
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
                torch.save(model_.state_dict(),"models/best_model.pth")
            else:
                patience_counter+=1
                if patience_counter>=patience:
                    st.warning("Early stopping triggered!")
                    try:
                        model_.load_state_dict(torch.load("models/best_model.pth"))
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

# Function to process video file and add bounding boxes
def process_video(video_path, model):
    all_detections = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join("runs", f"output_{os.path.basename(video_path)}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_inference(frame, model)

        for det in detections:
            x1, y1, x2, y2 = map(int, [det[0], det[1], det[2], det[3]])
            class_id = int(det[5])
            class_name = det[5]
            label = f"{class_name} {det[4]:.2f}"

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        if isinstance(detections, np.ndarray) and detections.size > 0:
            all_detections.append(detections)

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()

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
# Function for Training new YOLO Models 
def yolo_training():
    st.title("YOLO Model Training")
    st.subheader("Select Dataset Location and Update YAML")

    # Step 1: Dataset and YAML Selection
    yaml_file = st.file_uploader("Upload data.yaml", type="yaml")  # Upload data.yaml file

    if yaml_file:
        # Step 2: Save uploaded YAML file to the dataset folder in the current directory
        dataset_folder = os.path.join(os.getcwd(), "dataset")  # Get the current working directory and create a 'dataset' folder
        os.makedirs(dataset_folder, exist_ok=True)  # Ensure the dataset folder exists

        yaml_file_path = os.path.join(dataset_folder, "data.yaml")  # Define the path to save the YAML file

        # Save the uploaded file
        with open(yaml_file_path, "wb") as file:
            file.write(yaml_file.getbuffer())  # Write uploaded file content to the dataset folder

        try:
            # Step 3: Update YAML paths if necessary (add your function for path updates)
            update_yaml_paths(yaml_file_path, dataset_folder)
            st.session_state['dataset_folder'] = dataset_folder
            st.session_state['data_yaml_path'] = yaml_file_path

            # Step 4: Load and display classes from the YAML file (assuming you have a function for this)
            yolo_classes = load_classes_from_yaml(yaml_file_path)
            if yolo_classes:
                st.success(f"Classes loaded: {yolo_classes}")
            else:
                st.error("Error loading classes from YAML.")
        except Exception as e:
            st.error(f"An error occurred while updating the YAML file: {str(e)}")
    else:
        st.warning("Please upload the data.yaml file to proceed.")
        return

    dataset_path = st.session_state.get('dataset_folder')
    data_yaml_path = st.session_state.get('data_yaml_path')

    if dataset_path:
        # Step 4: Training Parameters
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

            # Correct the path to the 'train.py' file
            yolo_path = os.path.abspath("yolov5")
            train_py_path = os.path.join(yolo_path, "train.py")

            if not os.path.exists(train_py_path):
                st.error(f"train.py not found at {train_py_path}. Please check the YOLOv5 installation.")
                return

            # Prepare the training command
            command = [
                sys.executable, train_py_path,  # Use the absolute path for train.py
                '--img', str(img_size),
                '--batch', str(batch_size),
                '--epochs', str(epochs),
                '--data', data_yaml_path,
                '--weights', f'{yolo_model}.pt',
                '--device', '0'  # Assuming GPU, change to 'cpu' if necessary
            ]

            # Initialize progress bar and text display for output
            progress_bar = st.progress(0)
            output_box = st.empty()

            try:
                with st.spinner('Training in progress...'):
                    # Launch the process and read both stdout and stderr in real-time
                    process = subprocess.Popen(
                        command, cwd=yolo_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
                    )

                    total_lines = 0  # To simulate progress based on line output
                    for line in process.stdout:
                        if line.strip():  # Ensure the line is not empty
                            # Display each line in the Streamlit interface
                            output_box.text(line.strip())

                        # Simulate progress increment based on output lines
                        total_lines += 1
                        progress = min(total_lines / (epochs * 100), 1.0)  # Adjust this multiplier based on typical YOLO output length
                        progress_bar.progress(int(progress * 100))

                    process.stdout.close()
                    process.wait()

                if process.returncode == 0:
                    st.success("YOLO model trained successfully!")
                    progress_bar.progress(100)  # Set progress to 100% on completion
                else:
                    st.error("YOLO model training failed.")
                    st.write(process.stderr.read())  # Show the stderr output for debugging

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload a dataset and configure the YAML file.")

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
def run_transformer_statistics(model_path, df_detections, video_path=None):
    """
    Enhanced multi-label stats & GPT analysis for Transformer-based predictions.
    1) Re-chunk data => run inference.
    2) Merge chunk predictions => global frame-based predictions.
    3) Summarize time "resting" vs "active."
    4) Identify training (person+bag) vs sparring (two persons).
    5) GPT-based textual analysis of stats & trends.
    6) Display charts & optional video.
    """

    import numpy as np

    # Display CSV preview
    st.write("## CSV Preview")
    st.write(df_detections.head(30))
    st.write(f"CSV has {len(df_detections)} rows")
    st.write("Raw freq of class_id:", df_detections['class_id'].value_counts().to_dict())

    # Attempt to load the Transformer model
    model_ = load_transformer_model(
        model_path,
        d_model=64,          # or gather from user
        nhead=2,             # or gather from user
        num_layers=2,        # or gather from user
        dim_feedforward=128, # or gather from user
        dropout=0.1,
        num_classes=7
    )
    if model_ is None:
        st.error("Failed to load transformer.")
        return

    # Prepare data for the same chunk logic
    seq_len = 20   # or gather from user side
    stride  = 10
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

    model_.eval()
    with torch.no_grad():
        outs_list = []
        for i in range(len(inputs)):
            seq_in = inputs[i].unsqueeze(0).to(device)
            logits = model_(seq_in)
            outs_list.append(logits.cpu())
        all_logits = torch.cat(outs_list, dim=0)  # shape [N, seq_len, num_classes]

    # Convert to predictions
    probs  = torch.sigmoid(all_logits)
    preds  = (probs > 0.5).int().numpy()  # shape [N, seq_len, num_classes]

    # Merge chunk predictions => global_preds[frame] = [0/1 for each class]
    global_preds = {}
    N = len(inputs)
    for i in range(N):
        chunk_start = i*stride
        for t_ in range(seq_len):
            frame_idx = chunk_start + t_
            if frame_idx not in global_preds:
                global_preds[frame_idx] = [0]*num_classes
            for c_idx in range(num_classes):
                global_preds[frame_idx][c_idx] = max(global_preds[frame_idx][c_idx], preds[i,t_,c_idx])

    # Let's define a simple class_map
    class_map = {
        0:"boxing-bag",
        1:"cross",
        2:"high-guard",
        3:"hook",
        4:"kick",
        5:"low-guard",
        6:"person"
    }

    frames_sorted  = sorted(global_preds.keys())
    if not frames_sorted:
        st.warning("No frames in global predictions.")
        return

    # Summarize instance counts, durations, etc.
    # We'll also do "resting vs active" logic:
    # resting = low-guard
    # active = cross / hook / kick / high-guard

    instance_counts = [0]*num_classes
    durations       = [0]*num_classes
    prev_state      = [0]*num_classes
    log_lines       = []

    # For detecting # of person in each frame
    # if 2 => sparring, ignore boxing-bag
    # if 1 => training => we consider bag
    # We'll keep track how often 2 persons or 1 person appear
    training_frames = 0
    sparring_frames = 0

    for idx, frame_num in enumerate(frames_sorted):
        current = global_preds[frame_num]
        # Count how many "person" in the frame
        # Actually it's either 0 or 1, but let's assume the model can detect multiple persons -> sum for c=6
        # Typically c=6 => person
        # But if the model is strictly 0/1, we won't see 2. We'll assume your model can detect multiple or store it differently.
        # We'll do a naive approach: if "person" is 1 => there's at least one person
        # If 2 => you'd need a different detection approach. We'll do a placeholder
        # For now let's do: if global_preds says "person" => 1 person
        # We'll just do a "bag" => c=0, "person" => c=6
        num_person = current[6]  # 0 or 1 in this naive approach
        num_bag    = current[0]  # 0 or 1

        if num_person>1:
            # In your real logic, you might have to track bounding boxes, or the model might produce "person1," "person2."
            # We'll do a placeholder:
            sparring_frames += 1
        elif num_person==1 and num_bag==1:
            # training scenario
            training_frames += 1

        # We'll check transitions for each class
        for c_idx in range(num_classes):
            was_present = prev_state[c_idx]
            is_present  = current[c_idx]
            if was_present==0 and is_present==1:
                instance_counts[c_idx]+=1
                log_lines.append(f"Frame {frame_num}: New instance of '{class_map[c_idx]}' detected.")
            elif was_present==1 and is_present==0:
                log_lines.append(f"Frame {frame_num}: Instance of '{class_map[c_idx]}' ended.")

            # If present, accumulate duration
            if is_present==1:
                durations[c_idx]+=1

        prev_state=current

    # If last frame still has some classes as 1, we might have open intervals
    st.write("### Final Class Instance Counts")
    for c_idx in range(num_classes):
        st.write(f"{class_map[c_idx]}: {instance_counts[c_idx]}")

    # durations => number of frames
    max_frame = frames_sorted[-1]
    total_frames = max_frame+1
    st.write("### Class Durations (in frames and % of total):")
    for c_idx in range(num_classes):
        frac = (durations[c_idx]/total_frames)*100 if total_frames>0 else 0
        st.write(f"{class_map[c_idx]}: {durations[c_idx]} frames ({frac:.2f}%)")

    # Resting vs Active logic
    # resting => low-guard => index=5
    resting_frames = durations[5]
    # active => cross(1), hook(3), kick(4), high-guard(2)
    active_frames = durations[1]+durations[3]+durations[4]+durations[2]
    # We skip bag(0) and person(6) for active logic
    st.write("### Athlete Efficiency / Rest vs Active")
    st.write(f"Resting frames (low-guard): {resting_frames}")
    st.write(f"Active frames (punch/kick/guard): {active_frames}")
    # E.g. ratio
    rest_ratio  = (resting_frames/total_frames)*100 if total_frames>0 else 0
    active_ratio= (active_frames/total_frames)*100 if total_frames>0 else 0
    st.write(f"Rest Ratio:  {rest_ratio:.2f}%")
    st.write(f"Active Ratio:{active_ratio:.2f}%")

    # Summarize training vs sparring
    # We said: training_frames++ if person+bag in the frame
    # We said: sparring_frames++ if 2 persons
    # (You mentioned "two persons => ignore bag," but we only have a single 'person' class. This is conceptual.)
    # We'll just show them:
    st.write(f"Training frames (person+bag): {training_frames}")
    # st.write(f"Sparring frames (two persons) [conceptual]: {sparring_frames}")

    # Let's show a bar chart for durations
    # Build a dictionary or DataFrame
    import pandas as pd
    chart_data = pd.DataFrame({
        'Class': [class_map[i] for i in range(num_classes)],
        'DurationFrames': durations
    })
    chart_data.set_index('Class', inplace=True)
    st.bar_chart(chart_data['DurationFrames'])

    # Possibly a separate bar for instance_counts
    inst_data = pd.DataFrame({
        'Class': [class_map[i] for i in range(num_classes)],
        'Instances': instance_counts
    })
    inst_data.set_index('Class', inplace=True)
    st.bar_chart(inst_data['Instances'])

    # 7) GPT-based textual insights
    # Let's define a function to call GPT. We'll do a placeholder:
    def generate_gpt_analysis(stats_summary):
        """
        Example placeholder function to call GPT with stats_summary
        Real usage: pass an OpenAI API prompt or use something like:
        openai.Completion.create(engine=..., prompt=..., etc.)
        """
        # We'll just return a dummy text here, or you can do the actual API call
        # that references the stats_summary
        return f"GPT Analysis (placeholder): Based on these stats, we see {stats_summary}"

    # Let's compile a small text summary
    stats_str = (f"Instances: {instance_counts}, "
                 f"Durations: {durations}, "
                 f"Resting ratio: {rest_ratio:.2f}%, "
                 f"Active ratio: {active_ratio:.2f}%, "
                 f"Training frames: {training_frames}, Sparring frames: {sparring_frames}")
    # Call GPT
    gpt_result = generate_gpt_analysis(stats_str)
    st.write("### GPT Insights:")
    st.write(gpt_result)

    # 8) Optionally display video
    if video_path and os.path.exists(video_path):
        st.write("### Video Used:")
        st.video(video_path)
    else:
        st.write("No video path provided or file not found.")

#########################
#  RESULTS DASHBOARD
#########################
def transformer_results_dashboard():
    st.title("Results Dashboard (Transformer)")

    # pick model
    t_models=[f for f in os.listdir('models') if f.endswith('.pth')]
    sel_m=st.selectbox("Select Transformer Model", t_models)
    if sel_m:
        mp_=os.path.join("models", sel_m)
        st.success(f"Selected: {sel_m}")

        # pick CSV
        cfs=[f for f in os.listdir('runs') if f.endswith('.csv')]
        sel_c=st.selectbox("CSV File", cfs)
        if sel_c:
            cpath=os.path.join("runs", sel_c)
            st.success(f"Selected CSV: {sel_c}")

            df_=pd.read_csv(cpath)
            run_transformer_statistics(mp_, df_)

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
