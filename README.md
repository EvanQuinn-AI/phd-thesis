# 🥊 Combat Sports Automation

An AI-powered video analysis system for combat sports training, featuring real-time object detection, action recognition, and tactical insights using YOLOv11, Transformer models, and LLM analysis.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Overview

Combat Sports Automation is a comprehensive video analysis platform designed for combat sports training. The system automatically detects and analyzes training sessions, providing real-time statistics, action recognition, and AI-powered tactical insights.

### Key Capabilities

- **Real-time Object Detection**: Identifies boxing bags, persons, punches, kicks, and guard positions using YOLOv11
- **Action Recognition**: Tracks action sequences and transitions using Transformer models
- **Statistical Analysis**: Calculates punch/kick counts, active ratios, and performance metrics
- **AI-Powered Insights**: Generates tactical analysis using local LLM (Ollama) integration
- **Video Processing**: Annotates videos with bounding boxes and statistics
- **Model Training**: Interface for training custom YOLOv11 and Transformer models

## Features

- 🎯 **Real-time Video Analysis**: Process videos frame-by-frame with live statistics
- 📊 **Comprehensive Metrics**: Track punches, kicks, active ratio, and action sequences
- 🤖 **AI Tactical Analysis**: Get expert-level insights on training performance
- 📈 **Visual Analytics**: Action transition graphs, timeline visualizations, and statistics
- 🎬 **Video Annotation**: Download processed videos with bounding boxes and statistics
- 🔧 **Custom Model Training**: Train YOLOv11 and Transformer models on your own datasets
- ⚡ **GPU Acceleration**: Optimized for CUDA-enabled GPUs for faster processing
- 🎨 **Modern UI**: Professional dark-themed Streamlit interface

## Architecture

### Multi-Model System

The application employs a sophisticated multi-stage pipeline:

1. **YOLOv11 Object Detection** (Ultralytics)
   - Detects 6 classes: boxing-bag, high-guard, kick-knee, low-guard, person, punch
   - Real-time inference with GPU acceleration
   - Configurable confidence and IoU thresholds

2. **Transformer Action Recognition**
   - Processes frame sequences (32 frames) for action classification
   - Identifies action patterns and transitions
   - Builds action sequences for Markov analysis

3. **LLM Tactical Analysis** (Ollama)
   - Analyzes training statistics and action sequences
   - Generates tactical insights and coaching observations
   - Uses local LLM model for privacy and offline operation

### Pipeline Flow

```
Video Input
    ↓
[Frame Extraction]
    ↓
[YOLOv11 Detection] → Bounding Boxes (boxing-bag, person, punch, kick-knee, etc.)
    ↓
[Overlap Detection] → Action Events (punch/kick when overlapping with bag)
    ↓
[Frame Buffer] → 32-frame sequences
    ↓
[Transformer Analysis] → Action Recognition & Sequence Building
    ↓
[Statistics Calculation] → Counts, Ratios, Metrics
    ↓
[LLM Analysis] → Tactical Insights
    ↓
[Visualization & Export] → Graphs, Annotations, Processed Video
```

### Key Components

- **Real-time Processing**: Chunked frame processing to avoid Streamlit timeouts
- **Event Counting**: Temporal logic with gap tolerance for accurate action counting
- **State Management**: Session-based state for preserving analysis across reruns
- **Error Handling**: Robust error handling with graceful degradation
- **GPU Optimization**: Automatic device detection and optimization

## Project Structure

```
Combat Sports Automation/
├── experiment-version/
│   └── app.py              # Main Streamlit application (3,500+ lines)
│
├── dataset/                # Training dataset (YOLO format)
│   ├── train/
│   │   ├── images/         # Training images
│   │   └── labels/         # YOLO format labels
│   ├── valid/
│   │   ├── images/         # Validation images
│   │   └── labels/         # YOLO format labels
│   ├── test/
│   │   ├── images/         # Test images
│   │   └── labels/         # YOLO format labels
│   └── data.yaml           # Dataset configuration
│
├── models/                 # Trained model weights
│   ├── *.pt                # YOLOv11 model files (best.pt, yolo11*.pt)
│   └── transformer_model.pth  # Transformer model weights
│
├── data/                   # Input videos for analysis
│   └── *.mp4, *.avi, *.mov
│
├── runs/                   # Output directory
│   ├── *.mp4               # Processed videos with annotations
│   └── yolo_predictions_*.csv  # Detection results
│
├── requirements.txt        # Python dependencies
├── main.py                 # YOLO training script
├── run_experiment.bat      # Windows batch script to run app
└── README.md              # This file
```

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Optional, but recommended for GPU acceleration (CUDA 11.0+)
- **Ollama**: Required for AI tactical analysis (download from [ollama.ai](https://ollama.ai))

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd "Combat Sports Automation"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `ultralytics>=8.2.34` - YOLOv11 models
- `streamlit` - Web interface
- `torch>=1.8.0` - Deep learning framework
- `opencv-python>=4.1.1` - Computer vision
- `pandas>=1.1.4` - Data processing
- `numpy>=1.23.5` - Numerical computing
- `ollama` - LLM integration (install separately)

### Step 3: Install Ollama

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and start Ollama service
3. Pull the required model:
   ```bash
   ollama pull gpt-oss:120b-cloud
   ```

### Step 4: Setup Models

**Option A: Use Pretrained Models**
- Place YOLOv11 model files (`.pt`) in `models/` directory
- Ensure `best.pt` exists or model will auto-select newest `.pt` file

**Option B: Train Custom Models**
- Use the training interface in the app (see [Usage](#usage) section)
- Or use `main.py` for direct YOLOv11 training

## Setup

### 1. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print('YOLOv11: OK')"
python -c "import streamlit; print('Streamlit: OK')"
```

### 2. Check GPU Availability (Optional)

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 3. Verify Ollama Service

```bash
ollama list
```

Should show `gpt-oss:120b-cloud` in the list.

### 4. Run Application

```bash
streamlit run experiment-version/app.py
```

Or use the batch script (Windows):
```bash
run_experiment.bat
```

The application will open in your default browser at `http://localhost:8501`

## Usage

### Real-Time Video Analysis

1. **Upload Video**
   - Click "📤 Upload video file"
   - Select MP4, AVI, or MOV file
   - Wait for upload confirmation

2. **Start Analysis**
   - Click "▶️ Start Analysis" button
   - Video processes in real-time with:
     - Bounding boxes around detected objects
     - Real-time statistics (FPS, hits, punches, kicks)
     - Action detection indicators

3. **View Results**
   - After processing completes, view:
     - Final statistics (punch count, kick count, total hits, active ratio)
     - Action transition graphs
     - AI-powered tactical analysis

4. **Download Processed Video**
   - Click "🎬 Generate & Download" to create annotated video
   - Download includes bounding boxes and statistics overlay

### Training Custom Models

#### YOLOv11 Training

1. Navigate to "🎯 YOLO Model Training" (if available in interface)
2. Upload `data.yaml` file or use default in `dataset/` directory
3. Configure training parameters:
   - **Epochs**: Number of training iterations (default: 100)
   - **Batch Size**: Images per batch (default: 16)
   - **Image Size**: Input resolution (default: 640)
   - **Model**: Select YOLOv11 variant (n/s/m/l/x)
4. Click "🚀 Start YOLO Training"
5. Monitor training progress in console
6. Trained model saved to `models/` directory

**Alternative: Direct Training Script**

```bash
python main.py
```

Edit `main.py` to configure:
- Dataset path
- Model variant (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
- Training parameters

#### Transformer Training

1. Navigate to "🧠 Transformer Training"
2. Upload CSV file with frame-level detections (from YOLO predictions)
3. Configure model parameters
4. Click "🚀 Train Transformer"
5. Model saved as `models/transformer_model.pth`

### Understanding Output

#### Statistics Explained

- **Total Hits**: Sum of detected punches and kicks
- **Punches**: Number of punch events (detected when punch overlaps with bag)
- **Kicks/Knees**: Number of kick/knee events
- **Active Ratio**: Percentage of frames with active actions
- **FPS**: Processing frames per second

#### Action Sequences

The system tracks action sequences and builds transition graphs showing:
- Action patterns (punch → kick → guard)
- Transition probabilities
- Action duration statistics

#### AI Analysis

The LLM analysis provides:
- 5 tactical observations
- Performance insights
- Training recommendations
- Based on actual statistics (no fabricated data)

## Configuration

### Model Paths

Edit `app.py` to change default paths:

```python
# YOLO Model Selection (auto-selects best.pt or newest .pt file)
models_dir = "models"

# Transformer Model
transformer_model_path = "models/transformer_model.pth"
```

### Detection Thresholds

In `yolo_inference()` function:

```python
results = model(rgb_frame, conf=0.1, iou=0.2, verbose=False)
```

- `conf`: Confidence threshold (default: 0.1)
- `iou`: Intersection over Union threshold (default: 0.2)

### Event Counting Parameters

In `process_video()` and `process_realtime_frame()`:

```python
min_event_dur = {'punch': 2, 'kick-knee': 6}  # Minimum frames for event
gap_tolerance = {'punch': 1, 'kick-knee': 4}  # Gap tolerance in frames
CONF_THR = 0.4  # Confidence threshold for filtering
```

### Ollama Configuration

In `app.py`:

```python
ollama_model_name = "gpt-oss:120b-cloud"  # Change to your preferred model
```

### Chunked Processing

For long videos, frames are processed in chunks:

```python
max_frames_per_chunk = 30  # Frames per Streamlit execution
```

Adjust this value to balance between responsiveness and processing speed.

## Troubleshooting

### Common Issues

#### 1. "No YOLO models found"

**Solution:**
- Ensure `.pt` model files exist in `models/` directory
- Place `best.pt` or any YOLOv11 model file in `models/`
- Model will auto-select `best.pt` if available

#### 2. "Failed to load YOLO model"

**Solution:**
- Verify model file is valid YOLOv11 format
- Check file permissions
- Ensure ultralytics package is installed: `pip install ultralytics`

#### 3. Ollama Connection Error

**Solution:**
- Verify Ollama service is running: `ollama serve`
- Check if model is available: `ollama list`
- Pull model if missing: `ollama pull gpt-oss:120b-cloud`
- Verify connection: `curl http://localhost:11434/api/tags`

#### 4. GPU Not Detected

**Solution:**
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch if needed
- Application will fallback to CPU automatically

#### 5. Streamlit Timeout for Long Videos

**Solution:**
- Already handled with chunked processing (30 frames per execution)
- If issues persist, reduce `max_frames_per_chunk` in code
- Consider processing shorter video segments

#### 6. Memory Errors

**Solution:**
- Reduce batch size in training
- Process videos in smaller chunks
- Close other applications to free memory
- Use CPU mode if GPU memory is limited

#### 7. Index Out of Bounds Errors

**Solution:**
- Ensure dataset has all 6 classes defined in `data.yaml`
- Verify class IDs match between training and inference
- Check `yolo_classes` in session state matches model classes

### Getting Help

- Check console output for detailed error messages
- Review `app.py` comments for function documentation
- Verify all dependencies are installed correctly
- Check Streamlit logs for runtime errors

## Development

### Code Structure

**app.py** is organized into sections:

1. **Imports & Configuration** (Lines 1-44)
   - Library imports
   - Device configuration
   - Streamlit setup

2. **UI Components** (Lines 45-333)
   - CSS styling
   - Reusable UI components
   - Status cards and badges

3. **Helper Functions** (Lines 334-577)
   - Data collection utilities
   - YAML handling
   - Model loading and selection

4. **YOLO Functions** (Lines 416-658)
   - Model loading with GPU optimization
   - Inference functions
   - Video processing

5. **Transformer Functions** (Lines 1458-1721)
   - Model architecture
   - Training interface
   - Inference functions

6. **HMM & Analysis** (Lines 1722-2026)
   - State sequence compression
   - Transition graph creation
   - Metrics display

7. **Real-time Processing** (Lines 844-1394)
   - Frame processing
   - Statistics tracking
   - Event counting logic

8. **LLM Integration** (Lines 2494-2978)
   - Ollama connection
   - Analysis generation
   - Streaming responses

9. **Main Interface** (Lines 3059-3503)
   - Real-time analysis UI
   - Video upload and processing
   - Results display

### Key Functions Reference

#### `load_yolo_model(weights='models/best.pt')`
Loads YOLOv11 model with automatic GPU optimization and caching.

#### `yolo_inference(frame, model, max_size=640)`
Performs object detection on a single frame. Returns detections in format `[x1, y1, x2, y2, conf, cls]`.

#### `process_video(video_path, model)`
Processes entire video file, writes annotated output to `runs/` directory.

#### `process_realtime_frame(...)`
Processes single frame for real-time analysis, updates statistics and displays.

#### `generate_end_of_video_analysis(final_stats, action_sequence, analysis_queue)`
Generates AI tactical analysis using Ollama LLM with streaming support.

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Future Improvements

- [ ] Multi-person tracking
- [ ] Real-time camera feed support
- [ ] Advanced analytics dashboard
- [ ] Export to multiple video formats
- [ ] Mobile app integration
- [ ] Cloud deployment support
- [ ] Additional combat sports (MMA, Muay Thai, etc.)
- [ ] Performance benchmarking tools

## License

[Add your license information here]

## Acknowledgments

- [Ultralytics](https://ultralytics.com) for YOLOv11
- [Streamlit](https://streamlit.io) for web interface framework
- [Ollama](https://ollama.ai) for local LLM integration

---

**Built with ❤️ for combat sports training and analysis**

