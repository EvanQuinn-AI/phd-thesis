# app_pose_enhanced_v2.py
# Enhanced Combat Sports Analysis with Scientific Metrics
# - Accurate pose detection with velocity analysis
# - Optimized UI streaming (output only)
# - Real-time velocity visualization
# - Comprehensive scientific statistics

import os
import json
import logging
import warnings
import time
import pandas as pd
import numpy as np
import torch
import yaml
import streamlit as st
import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
from pose_strike_detector_enhanced import PoseStrikeDetector

# Suppress warnings
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
torch.classes.__path__ = []
warnings.filterwarnings("ignore", category=FutureWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
logging.getLogger("torch").setLevel(logging.ERROR)
st.set_page_config(page_title="Combat Sports - Enhanced Analysis", layout="wide")


# =============================================================================
# YOLO UTILITIES
# =============================================================================

def load_yolo_model(model_path: str):
    """Load YOLO model (supports both YOLOv5 and YOLOv8)
    
    Returns:
        tuple: (model, model_type) where model_type is 'v5' or 'v8'
    """
    try:
        # First try YOLOv8 (Ultralytics)
        from ultralytics import YOLO
        model = YOLO(model_path)
        st.info("✅ Loaded with Ultralytics (YOLOv8+)")
        return model, 'v8'
    except Exception as e:
        if "YOLOv5" in str(e) or "forwards compatible" in str(e):
            # This is a YOLOv5 model, load with torch.hub
            try:
                st.warning("⚠️ Detected YOLOv5 model. Loading with torch.hub...")
                import torch
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
                st.info("✅ Loaded YOLOv5 model successfully")
                return model, 'v5'
            except Exception as e2:
                st.error(f"Failed to load YOLOv5 model: {e2}")
                st.info("💡 Try: pip install yolov5")
                return None, None
        else:
            st.error(f"Failed to load YOLO: {e}")
            return None, None


def load_classes_from_yaml(yaml_path: str) -> List[str]:
    """Load class names from data.yaml"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    except Exception as e:
        st.warning(f"Could not load class names: {e}")
        return []


def merge_overlapping_boxes(boxes: List, iou_thresh: float = 0.5) -> List:
    """Merge overlapping bounding boxes"""
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence
    merged = []
    
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    while boxes:
        box = boxes.pop(0)
        merged.append(box)
        boxes = [b for b in boxes if iou(box, b) < iou_thresh]
    
    return merged


# =============================================================================
# VIDEO PROCESSING WITH ENHANCED POSE DETECTION
# =============================================================================

def process_video_enhanced(video_path: str,
                          yolo_model,
                          yolo_model_type: str,  # 'v5' or 'v8'
                          class_names: List[str],
                          pose_params: Dict,
                          enable_yolo: bool = True,
                          enable_pose: bool = True,
                          stream_output: bool = True) -> Tuple[List[Dict], str, Dict]:
    """
    Process video with enhanced pose detection and scientific metrics.
    
    Args:
        video_path: Path to input video
        yolo_model: YOLO model instance
        class_names: List of class names
        pose_params: Pose detector parameters
        enable_yolo: Enable YOLO detection
        enable_pose: Enable pose detection
        stream_output: Stream output video to UI (not input)
        
    Returns:
        (frame_data, output_path, scientific_metrics)
    """
    # Setup
    os.makedirs("runs", exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"runs/{base}_enhanced.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return [], "", {}
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    # Initialize pose detector
    pose_detector = None
    if enable_pose:
        with st.spinner("Initializing pose detector..."):
            pose_detector = PoseStrikeDetector(**pose_params)
            pose_detector.initialize()
    
    # UI placeholders (output only)
    if stream_output:
        st.subheader("📹 Processing Output")
        col1, col2 = st.columns([3, 1])
        with col1:
            output_frame_placeholder = st.empty()
        with col2:
            stats_placeholder = st.empty()
            velocity_chart_placeholder = st.empty()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Processing state
    frame_idx = 0
    all_frame_data = []
    
    # YOLO state
    yolo_counters = {'punch': 0, 'kick': 0}
    yolo_in_event = {'punch': False, 'kick': False}
    yolo_event_start = {'punch': 0, 'kick': 0}
    yolo_gap = {'punch': 0, 'kick': 0}
    
    # Velocity tracking for real-time chart
    velocity_history = {
        'frames': deque(maxlen=100),
        'wrist': deque(maxlen=100),
        'ankle': deque(maxlen=100),
        'knee': deque(maxlen=100),
    }
    
    # Processing loop
    st.info(f"🎬 Processing {total_frames} frames at {fps:.1f} fps...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp for pose detection
        timestamp_ms = int(frame_idx * 1000 / fps)
        
        # Initialize flags
        ov_punch = False
        ov_kick = False
        bag_boxes = []
        
        # YOLO detection
        if enable_yolo and yolo_model:
            # Handle YOLOv5 vs YOLOv8 inference
            if yolo_model_type == 'v8':
                # YOLOv8/YOLO11 - supports verbose parameter
                results = yolo_model(frame, verbose=False)
                if results and len(results[0].boxes) > 0:
                    detections = results[0].boxes.data.cpu().numpy()
                else:
                    detections = np.array([])
            else:
                # YOLOv5 - no verbose parameter
                results = yolo_model(frame)
                if results is not None and len(results.xyxy[0]) > 0:
                    detections = results.xyxy[0].cpu().numpy()
                else:
                    detections = np.array([])
            
            if len(detections) > 0:
                
                # Filter detections
                filtered = []
                raw_bag = []
                
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
                    cls_id = int(cls_id)
                    
                    if conf < 0.25:
                        continue
                    
                    # Collect bag boxes for pose detection
                    if cls_id == 0:  # bag
                        raw_bag.append([x1, y1, x2, y2, conf, cls_id])
                    
                    # Collect strike detections
                    if cls_id in [5, 2]:  # punch, kick-knee
                        filtered.append(det)
                
                # Merge overlapping bags
                bag_boxes = merge_overlapping_boxes(raw_bag)
                
                # Check overlaps with person (class 4)
                person_boxes = [det for det in detections if int(det[5]) == 4 and det[4] >= 0.25]
                
                for strike_det in filtered:
                    sx1, sy1, sx2, sy2, _, strike_cls = strike_det
                    strike_cls = int(strike_cls)
                    
                    for person_det in person_boxes:
                        px1, py1, px2, py2, _, _ = person_det
                        
                        # Check overlap
                        x_overlap = max(0, min(sx2, px2) - max(sx1, px1))
                        y_overlap = max(0, min(sy2, py2) - max(sy1, py1))
                        overlap = x_overlap * y_overlap
                        
                        strike_area = (sx2 - sx1) * (sy2 - sy1)
                        overlap_ratio = overlap / strike_area if strike_area > 0 else 0
                        
                        if overlap_ratio > 0.3:
                            if strike_cls == 5:  # punch
                                ov_punch = True
                            elif strike_cls == 2:  # kick-knee
                                ov_kick = True
                            break
                
                # Event counting
                for action, is_over in [('punch', ov_punch), ('kick', ov_kick)]:
                    if is_over:
                        yolo_gap[action] = 0
                        if not yolo_in_event[action]:
                            yolo_in_event[action] = True
                            yolo_event_start[action] = frame_idx
                    else:
                        if yolo_in_event[action]:
                            yolo_gap[action] += 1
                            if yolo_gap[action] >= 2:
                                dur = frame_idx - yolo_event_start[action]
                                if dur >= 2:
                                    yolo_counters[action] += 1
                                yolo_in_event[action] = False
                                yolo_gap[action] = 0
                
                # Draw YOLO boxes
                for det in filtered:
                    x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
                    cls_id = int(cls_id)
                    color = {0: (0,255,0), 5: (255,0,255), 2: (0,255,255), 4: (255,0,0)}.get(cls_id, (128,128,128))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    if cls_id < len(class_names):
                        cv2.putText(frame, f"{class_names[cls_id]} {conf:.2f}", (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Pose detection
        pose_result = None
        if enable_pose and pose_detector is not None:
            pose_result = pose_detector.process_frame(frame, bag_boxes, frame_idx, timestamp_ms)
            frame = pose_detector.draw_overlay(frame, pose_result, draw_velocity=True)
        
        # Draw comprehensive stats overlay on frame
        # Background panel
        cv2.rectangle(frame, (5, 5), (350, 230), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (350, 230), (255, 255, 255), 2)
        
        y_pos = 30
        cv2.putText(frame, "YOLO DETECTION", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y_pos += 30
        cv2.putText(frame, f"  Punches: {yolo_counters['punch']}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y_pos += 25
        cv2.putText(frame, f"  Kicks: {yolo_counters['kick']}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if pose_detector:
            y_pos += 35
            cv2.putText(frame, "POSE DETECTION", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            y_pos += 30
            cv2.putText(frame, f"  Punches: {pose_detector.punch_count}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
            y_pos += 25
            cv2.putText(frame, f"  Kicks: {pose_detector.kick_count}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
            y_pos += 25
            cv2.putText(frame, f"  Knees: {pose_detector.knee_count}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Velocity display
            if pose_result and pose_result.get('velocities'):
                vel = pose_result['velocities']
                y_pos += 30
                max_wrist = max(vel.get('left_wrist', 0), vel.get('right_wrist', 0))
                max_ankle = max(vel.get('left_ankle', 0), vel.get('right_ankle', 0))
                cv2.putText(frame, f"Wrist: {max_wrist:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 200), 1)
                y_pos += 20
                cv2.putText(frame, f"Ankle: {max_ankle:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
        
        # Write frame
        out.write(frame)
        
        # Collect frame data
        frame_info = {
            'frame': frame_idx,
            'timestamp': frame_idx / fps,
            'yolo_punch_hit': ov_punch,
            'yolo_kick_hit': ov_kick,
            'yolo_punch_count': yolo_counters['punch'],
            'yolo_kick_count': yolo_counters['kick'],
        }
        
        if pose_result:
            frame_info.update({
                'pose_punch_hit': pose_result.get('pose_punch_hit', False),
                'pose_kick_hit': pose_result.get('pose_kick_hit', False),
                'pose_knee_hit': pose_result.get('pose_knee_hit', False),
                'pose_punch_count': pose_result.get('punch_count', 0),
                'pose_kick_count': pose_result.get('kick_count', 0),
                'pose_knee_count': pose_result.get('knee_count', 0),
            })
            
            # Velocity data
            vel = pose_result.get('velocities', {})
            frame_info['wrist_vel_max'] = max(vel.get('left_wrist', 0), vel.get('right_wrist', 0))
            frame_info['ankle_vel_max'] = max(vel.get('left_ankle', 0), vel.get('right_ankle', 0))
            frame_info['knee_vel_max'] = max(vel.get('left_knee', 0), vel.get('right_knee', 0))
            
            # Update velocity history for real-time chart
            velocity_history['frames'].append(frame_idx)
            velocity_history['wrist'].append(frame_info['wrist_vel_max'])
            velocity_history['ankle'].append(frame_info['ankle_vel_max'])
            velocity_history['knee'].append(frame_info['knee_vel_max'])
        
        all_frame_data.append(frame_info)
        
        # Stream output to UI (every 5 frames for performance)
        if stream_output and frame_idx % 5 == 0:
            with col1:
                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output_frame_placeholder.image(display, channels="RGB", use_container_width=True)
            
            with col2:
                # Stats
                stats_md = f"""
                **Frame:** {frame_idx}/{total_frames}
                
                **YOLO:**
                - Punch: {yolo_counters['punch']}
                - Kick: {yolo_counters['kick']}
                """
                
                if pose_detector:
                    stats_md += f"""
                **Pose:**
                - Punch: {pose_detector.punch_count}
                - Kick: {pose_detector.kick_count}
                - Knee: {pose_detector.knee_count}
                """
                
                stats_placeholder.markdown(stats_md)
                
                # Real-time velocity chart
                if len(velocity_history['frames']) > 10:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(velocity_history['frames']),
                        y=list(velocity_history['wrist']),
                        mode='lines',
                        name='Wrist',
                        line=dict(color='rgb(255, 100, 255)', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(velocity_history['frames']),
                        y=list(velocity_history['ankle']),
                        mode='lines',
                        name='Ankle',
                        line=dict(color='rgb(100, 255, 255)', width=2)
                    ))
                    fig.update_layout(
                        title="Velocity",
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=True,
                        legend=dict(x=0, y=1)
                    )
                    velocity_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"vel_{frame_idx}")
            
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            status_text.text(f"Processing: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Get scientific metrics
    scientific_metrics = {}
    if pose_detector:
        scientific_metrics = pose_detector.get_scientific_metrics()
        pose_detector.close()
    
    if stream_output:
        progress_bar.empty()
        status_text.empty()
    
    # Save CSV
    df = pd.DataFrame(all_frame_data)
    csv_path = f"runs/{base}_enhanced_data.csv"
    df.to_csv(csv_path, index=False)
    
    st.success(f"""
    ✅ **Processing Complete!**
    
    **Final Counts:**
    - YOLO: Punch={yolo_counters['punch']}, Kick={yolo_counters['kick']}
    - Pose: Punch={pose_detector.punch_count if pose_detector else 0}, Kick={pose_detector.kick_count if pose_detector else 0}, Knee={pose_detector.knee_count if pose_detector else 0}
    
    📁 Data: `{csv_path}`
    🎥 Video: `{out_path}`
    """)
    
    return all_frame_data, out_path, scientific_metrics


# =============================================================================
# SCIENTIFIC STATISTICS VISUALIZATION
# =============================================================================

def display_scientific_statistics(frame_data: List[Dict], metrics: Dict):
    """Display comprehensive scientific statistics"""
    
    st.header("📊 Scientific Analysis")
    
    df = pd.DataFrame(frame_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", len(df))
        st.metric("YOLO Punches", df['yolo_punch_count'].max())
    
    with col2:
        duration = df['timestamp'].max()
        st.metric("Duration (s)", f"{duration:.1f}")
        st.metric("YOLO Kicks", df['yolo_kick_count'].max())
    
    with col3:
        if 'pose_punch_count' in df.columns:
            st.metric("Pose Punches", df['pose_punch_count'].max())
            st.metric("Pose Kicks", df['pose_kick_count'].max())
    
    with col4:
        if 'pose_knee_count' in df.columns:
            st.metric("Pose Knees", df['pose_knee_count'].max())
            if 'pose_punch_count' in df.columns:
                total_pose = df['pose_punch_count'].max() + df['pose_kick_count'].max() + df['pose_knee_count'].max()
                st.metric("Total Pose Strikes", total_pose)
    
    st.divider()
    
    # Velocity analysis
    if 'wrist_vel_max' in df.columns:
        st.subheader("🚀 Velocity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Velocity over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['wrist_vel_max'],
                mode='lines',
                name='Wrist (Punches)',
                line=dict(color='rgb(255, 100, 255)', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['ankle_vel_max'],
                mode='lines',
                name='Ankle (Kicks)',
                line=dict(color='rgb(100, 255, 255)', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['knee_vel_max'],
                mode='lines',
                name='Knee',
                line=dict(color='rgb(100, 255, 100)', width=1)
            ))
            
            # Mark strike events
            pose_punches = df[df['pose_punch_hit'] == True]
            if len(pose_punches) > 0:
                fig.add_trace(go.Scatter(
                    x=pose_punches['timestamp'],
                    y=pose_punches['wrist_vel_max'],
                    mode='markers',
                    name='Punch Detected',
                    marker=dict(color='red', size=10, symbol='star')
                ))
            
            pose_kicks = df[df['pose_kick_hit'] == True]
            if len(pose_kicks) > 0:
                fig.add_trace(go.Scatter(
                    x=pose_kicks['timestamp'],
                    y=pose_kicks['ankle_vel_max'],
                    mode='markers',
                    name='Kick Detected',
                    marker=dict(color='orange', size=10, symbol='star')
                ))
            
            fig.update_layout(
                title="Velocity Over Time",
                xaxis_title="Time (s)",
                yaxis_title="Velocity (normalized)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Velocity distributions
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Wrist Velocity", "Ankle Velocity", "Knee Velocity"),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Histogram(x=df['wrist_vel_max'], nbinsx=50, name='Wrist',
                           marker_color='rgb(255, 100, 255)'),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=df['ankle_vel_max'], nbinsx=50, name='Ankle',
                           marker_color='rgb(100, 255, 255)'),
                row=2, col=1
            )
            fig.add_trace(
                go.Histogram(x=df['knee_vel_max'], nbinsx=50, name='Knee',
                           marker_color='rgb(100, 255, 100)'),
                row=3, col=1
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Velocity Distributions")
            fig.update_xaxes(title_text="Velocity", row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Strike timing analysis
    if 'pose_punch_count' in df.columns:
        st.subheader("⏱️ Strike Timing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative strikes over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['yolo_punch_count'],
                mode='lines',
                name='YOLO Punches',
                line=dict(color='rgb(255, 0, 255)', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['pose_punch_count'],
                mode='lines',
                name='Pose Punches',
                line=dict(color='rgb(255, 100, 255)', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['yolo_kick_count'],
                mode='lines',
                name='YOLO Kicks',
                line=dict(color='rgb(0, 255, 255)', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['pose_kick_count'],
                mode='lines',
                name='Pose Kicks',
                line=dict(color='rgb(100, 255, 255)', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Cumulative Strike Count",
                xaxis_title="Time (s)",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Comparison bar chart
            comparison_data = {
                'Method': ['YOLO', 'YOLO', 'Pose', 'Pose', 'Pose'],
                'Strike Type': ['Punch', 'Kick', 'Punch', 'Kick', 'Knee'],
                'Count': [
                    df['yolo_punch_count'].max(),
                    df['yolo_kick_count'].max(),
                    df['pose_punch_count'].max(),
                    df['pose_kick_count'].max(),
                    df['pose_knee_count'].max() if 'pose_knee_count' in df.columns else 0
                ]
            }
            comp_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(comp_df, x='Strike Type', y='Count', color='Method',
                        barmode='group',
                        color_discrete_map={'YOLO': 'rgb(100, 100, 255)', 'Pose': 'rgb(100, 255, 100)'})
            fig.update_layout(
                title="YOLO vs Pose Detection Comparison",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Metrics from enhanced detector
    if metrics:
        st.subheader("📈 Advanced Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Punch Statistics**")
            if 'punch_velocity_mean' in metrics:
                st.write(f"Mean Velocity: {metrics['punch_velocity_mean']:.4f}")
                st.write(f"Std Dev: {metrics['punch_velocity_std']:.4f}")
                st.write(f"Max Velocity: {metrics['punch_velocity_max']:.4f}")
        
        with col2:
            st.markdown("**Kick Statistics**")
            if 'kick_velocity_mean' in metrics:
                st.write(f"Mean Velocity: {metrics['kick_velocity_mean']:.4f}")
                st.write(f"Std Dev: {metrics['kick_velocity_std']:.4f}")
                st.write(f"Max Velocity: {metrics['kick_velocity_max']:.4f}")
        
        with col3:
            st.markdown("**Knee Statistics**")
            if 'knee_velocity_mean' in metrics:
                st.write(f"Mean Velocity: {metrics['knee_velocity_mean']:.4f}")
                st.write(f"Std Dev: {metrics['knee_velocity_std']:.4f}")
                st.write(f"Max Velocity: {metrics['knee_velocity_max']:.4f}")


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.title("🥊 Combat Sports Analysis - Enhanced")
    st.markdown("""
    **Enhanced pose detection** with velocity analysis, directional filtering, and acceleration detection.
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # YOLO model
    yolo_models = []
    if os.path.exists('models'):
        yolo_models = [f for f in os.listdir('models') if f.endswith('.pt')]
    
    model_path = None
    if yolo_models:
        selected = st.sidebar.selectbox("YOLO Model", yolo_models)
        model_path = os.path.join('models', selected)
    else:
        st.sidebar.warning("⚠️ No YOLO models in 'models/'")
    
    # Class names
    yaml_path = os.path.join("dataset", "data.yaml")
    class_names = load_classes_from_yaml(yaml_path) if os.path.exists(yaml_path) else \
                  ['bag', 'high-guard', 'kick-knee', 'low-guard', 'person', 'punch']
    
    # Detection options
    st.sidebar.subheader("Detection Options")
    enable_yolo = st.sidebar.checkbox("✅ YOLO Detection", value=True)
    enable_pose = st.sidebar.checkbox("✅ Enhanced Pose Detection", value=True)
    stream_output = st.sidebar.checkbox("📺 Stream Output Video", value=True,
                                       help="Stream processed video in real-time (reduces performance)")
    
    # Enhanced pose parameters
    with st.sidebar.expander("🎯 Pose Detection Tuning", expanded=False):
        st.markdown("**Punch Detection**")
        punch_vel_min = st.slider("Min Punch Velocity", 0.01, 0.15, 0.05, 0.005)
        punch_vel_max = st.slider("Max Punch Velocity", 0.15, 0.5, 0.3, 0.01)
        punch_accel = st.slider("Punch Decel Threshold", 0.005, 0.05, 0.02, 0.005)
        min_arm = st.slider("Min Arm Extension", 0.2, 0.9, 0.4, 0.05)
        
        st.markdown("**Kick Detection**")
        kick_vel_min = st.slider("Min Kick Velocity", 0.01, 0.15, 0.06, 0.005)
        kick_vel_max = st.slider("Max Kick Velocity", 0.15, 0.5, 0.35, 0.01)
        kick_accel = st.slider("Kick Decel Threshold", 0.005, 0.05, 0.025, 0.005)
        min_leg = st.slider("Min Leg Extension", 0.2, 0.9, 0.4, 0.05)
        
        st.markdown("**General**")
        cooldown = st.slider("Cooldown Frames", 5, 30, 15)
        target_margin = st.slider("Target Margin", 0.01, 0.1, 0.03, 0.01)
    
    pose_params = {
        'punch_vel_min': punch_vel_min,
        'punch_vel_max': punch_vel_max,
        'punch_accel_thresh': punch_accel,
        'min_arm_ext': min_arm,
        'kick_vel_min': kick_vel_min,
        'kick_vel_max': kick_vel_max,
        'kick_accel_thresh': kick_accel,
        'min_leg_ext': min_leg,
        'cooldown': cooldown,
        'target_margin': target_margin,
    }
    
    # Main content
    st.header("📤 Upload Video")
    video_file = st.file_uploader("Select training video", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        os.makedirs("data", exist_ok=True)
        video_path = os.path.join("data", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Show input video info (but don't stream it)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(video_file)
        with col2:
            st.info(f"""
            **Video Info:**
            - Name: {video_file.name}
            - Size: {video_file.size / 1024 / 1024:.1f} MB
            
            Click **Run Analysis** to process.
            """)
        
        if st.button("🚀 Run Enhanced Analysis", type="primary"):
            yolo_model = None
            yolo_model_type = None
            if enable_yolo and model_path:
                with st.spinner("Loading YOLO model..."):
                    yolo_model, yolo_model_type = load_yolo_model(model_path)
                    if yolo_model:
                        # Move to device (v8 supports .to(), v5 handles it differently)
                        if yolo_model_type == 'v8':
                            yolo_model.to(device)
                        else:
                            # YOLOv5 model already on correct device via torch.hub
                            pass
            
            # Process video
            frame_data, out_path, scientific_metrics = process_video_enhanced(
                video_path, yolo_model, yolo_model_type, class_names, pose_params,
                enable_yolo, enable_pose, stream_output
            )
            
            # Display scientific statistics
            if frame_data:
                display_scientific_statistics(frame_data, scientific_metrics)
            
            # Download buttons
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if out_path and os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "📥 Download Processed Video",
                            f.read(),
                            os.path.basename(out_path),
                            "video/mp4",
                            use_container_width=True
                        )
            
            with col2:
                csv_path = out_path.replace('.mp4', '_data.csv').replace('_enhanced', '_enhanced_data')
                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            "📊 Download CSV Data",
                            f.read(),
                            os.path.basename(csv_path),
                            "text/csv",
                            use_container_width=True
                        )


if __name__ == "__main__":
    main()
