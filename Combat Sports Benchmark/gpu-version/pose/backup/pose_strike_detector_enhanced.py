# pose_strike_detector_enhanced.py
# Enhanced MediaPipe Tasks API version with velocity analysis and scientific metrics
# Addresses accuracy issues through directional filtering and acceleration analysis

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import urllib.request
import os

# =============================================================================
# MODEL DOWNLOAD HELPER
# =============================================================================

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
POSE_MODEL_PATH = "pose_landmarker.task"

def download_pose_model(model_path: str = POSE_MODEL_PATH) -> str:
    """Download the pose landmarker model if not present"""
    if not os.path.exists(model_path):
        print(f"📥 Downloading pose model to {model_path}...")
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
        print("✅ Download complete!")
    return model_path


# =============================================================================
# POSE LANDMARK INDICES
# =============================================================================

LANDMARKS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]


# =============================================================================
# ENHANCED POSE STRIKE DETECTOR
# =============================================================================

class PoseStrikeDetector:
    """
    Enhanced strike detection with velocity analysis and directional filtering.
    
    Improvements over basic version:
    - Velocity direction check (must move towards target)
    - Acceleration analysis (detects impact moment)
    - Min/Max velocity bounds (filters noise and tracking errors)
    - Separate thresholds for different strike types
    - Scientific metrics collection for analysis
    """
    
    def __init__(self, 
                 # Punch parameters (tighter thresholds for accuracy)
                 punch_vel_min: float = 0.05,      # Increased from 0.03
                 punch_vel_max: float = 0.3,       # Max to filter tracking errors
                 punch_accel_thresh: float = 0.02,  # Detect deceleration on impact
                 min_arm_ext: float = 0.4,         # Reduced to catch hooks/uppercuts
                 
                 # Kick parameters
                 kick_vel_min: float = 0.06,       # Increased from 0.04
                 kick_vel_max: float = 0.35,
                 kick_accel_thresh: float = 0.025,
                 min_leg_ext: float = 0.4,         # Reduced for close-range knees
                 
                 # Detection parameters
                 cooldown: int = 15,               # Increased from 10 (0.5s at 30fps)
                 target_margin: float = 0.03,      # Reduced from 0.05
                 history_length: int = 7):         # Increased for better velocity calc
        
        # Store parameters
        self.punch_vel_min = punch_vel_min
        self.punch_vel_max = punch_vel_max
        self.punch_accel_thresh = punch_accel_thresh
        self.min_arm_ext = min_arm_ext
        
        self.kick_vel_min = kick_vel_min
        self.kick_vel_max = kick_vel_max
        self.kick_accel_thresh = kick_accel_thresh
        self.min_leg_ext = min_leg_ext
        
        self.cooldown_frames = cooldown
        self.target_margin = target_margin
        
        # Landmarker
        self.landmarker = None
        
        # Position histories (longer history for better velocity calculation)
        self.history = {
            'left_wrist': deque(maxlen=history_length),
            'right_wrist': deque(maxlen=history_length),
            'left_ankle': deque(maxlen=history_length),
            'right_ankle': deque(maxlen=history_length),
            'left_knee': deque(maxlen=history_length),
            'right_knee': deque(maxlen=history_length),
        }
        
        # Velocity histories for acceleration calculation
        self.vel_history = {k: deque(maxlen=5) for k in self.history.keys()}
        
        # Strike counters
        self.punch_count = 0
        self.kick_count = 0
        self.knee_count = 0
        
        # Cooldown trackers
        self.cooldown = {
            'punch_l': 0, 'punch_r': 0,
            'kick_l': 0, 'kick_r': 0,
            'knee_l': 0, 'knee_r': 0
        }
        
        # Scientific metrics collection
        self.strike_metrics = []
        self.velocity_data = {
            'wrist': [], 'ankle': [], 'knee': []
        }
    
    def initialize(self, running_mode: str = 'VIDEO') -> None:
        """Initialize the pose landmarker"""
        download_pose_model(POSE_MODEL_PATH)
        
        base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        mode = vision.RunningMode.VIDEO if running_mode == 'VIDEO' else vision.RunningMode.IMAGE
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
    
    def close(self) -> None:
        """Release resources"""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
    
    def reset(self) -> None:
        """Reset all counters and state"""
        for h in self.history.values():
            h.clear()
        for h in self.vel_history.values():
            h.clear()
        self.punch_count = self.kick_count = self.knee_count = 0
        self.cooldown = {k: 0 for k in self.cooldown}
        self.strike_metrics = []
        self.velocity_data = {'wrist': [], 'ankle': [], 'knee': []}
    
    def _get_pos(self, landmarks, idx: int, vis_thresh: float = 0.5) -> Optional[Tuple[float, float]]:
        """Get normalized position of landmark"""
        if landmarks is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        if hasattr(lm, 'visibility') and lm.visibility < vis_thresh:
            return None
        return (lm.x, lm.y)
    
    def _velocity(self, history: deque) -> float:
        """Calculate instantaneous velocity from position history"""
        positions = [p for p in history if p is not None]
        if len(positions) < 2:
            return 0.0
        
        # Use only recent frames for instantaneous velocity
        recent = positions[-3:] if len(positions) >= 3 else positions[-2:]
        
        total = sum(
            np.sqrt((recent[i][0] - recent[i-1][0])**2 + 
                   (recent[i][1] - recent[i-1][1])**2)
            for i in range(1, len(recent))
        )
        return total / (len(recent) - 1)
    
    def _acceleration(self, vel_history: deque) -> float:
        """Calculate acceleration (change in velocity)"""
        if len(vel_history) < 2:
            return 0.0
        velocities = list(vel_history)
        # Negative acceleration indicates deceleration (impact)
        return velocities[-1] - velocities[-2]
    
    def _direction_vector(self, history: deque) -> Optional[Tuple[float, float]]:
        """Calculate direction of movement"""
        positions = [p for p in history if p is not None]
        if len(positions) < 2:
            return None
        
        # Direction from second-to-last to last position
        start, end = positions[-2], positions[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        magnitude = np.sqrt(dx**2 + dy**2)
        
        if magnitude < 1e-6:
            return None
        
        return (dx / magnitude, dy / magnitude)
    
    def _moving_towards_target(self, limb_pos: Tuple[float, float],
                               direction: Tuple[float, float],
                               target_boxes: List[Tuple[int, int, int, int]],
                               w: int, h: int) -> bool:
        """Check if limb is moving towards any target"""
        if not target_boxes or direction is None:
            return False
        
        for box in target_boxes:
            # Target center in normalized coordinates
            target_x = ((box[0] + box[2]) / 2) / w
            target_y = ((box[1] + box[3]) / 2) / h
            
            # Vector from limb to target
            to_target_x = target_x - limb_pos[0]
            to_target_y = target_y - limb_pos[1]
            
            # Normalize
            mag = np.sqrt(to_target_x**2 + to_target_y**2)
            if mag < 1e-6:
                continue
            
            to_target_x /= mag
            to_target_y /= mag
            
            # Dot product: > 0.5 means moving roughly towards target (within ~60°)
            dot = direction[0] * to_target_x + direction[1] * to_target_y
            if dot > 0.5:
                return True
        
        return False
    
    def _extension(self, p1, p2, p3) -> float:
        """Calculate limb extension (0=bent, 1=straight)"""
        if any(p is None for p in [p1, p2, p3]):
            return 0.0
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        angle = np.arccos(np.clip(np.dot(v1/n1, v2/n2), -1, 1))
        return angle / np.pi
    
    def _in_box(self, pos, box, w, h) -> bool:
        """Check if point is in bounding box (with margin)"""
        if pos is None:
            return False
        nx1 = box[0]/w - self.target_margin
        ny1 = box[1]/h - self.target_margin
        nx2 = box[2]/w + self.target_margin
        ny2 = box[3]/h + self.target_margin
        return nx1 <= pos[0] <= nx2 and ny1 <= pos[1] <= ny2
    
    def process_frame(self, frame: np.ndarray,
                      target_boxes: List[Tuple[int,int,int,int]],
                      frame_idx: int,
                      timestamp_ms: int) -> Dict[str, Any]:
        """
        Process frame and detect strikes with enhanced accuracy.
        
        Returns:
            Dictionary with detection results and metrics
        """
        h, w = frame.shape[:2]
        
        result = {
            'frame': frame_idx,
            'pose_detected': False,
            'pose_punch_hit': False,
            'pose_kick_hit': False,
            'pose_knee_hit': False,
            'punch_count': self.punch_count,
            'kick_count': self.kick_count,
            'knee_count': self.knee_count,
            'landmarks': None,
            'positions': {},
            'velocities': {},
            'accelerations': {},
            'extensions': {},
        }
        
        # Run pose detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not detection.pose_landmarks:
            return result
        
        result['pose_detected'] = True
        landmarks = detection.pose_landmarks[0]
        result['landmarks'] = landmarks
        
        # Extract positions
        pos = {name: self._get_pos(landmarks, idx) for name, idx in LANDMARKS.items()}
        result['positions'] = pos
        
        # Update position histories
        for key in self.history:
            self.history[key].append(pos.get(key))
        
        # Calculate velocities
        vel = {k: self._velocity(self.history[k]) for k in self.history}
        result['velocities'] = vel
        
        # Update velocity histories and calculate accelerations
        accel = {}
        for key in self.vel_history:
            self.vel_history[key].append(vel[key])
            accel[key] = self._acceleration(self.vel_history[key])
        result['accelerations'] = accel
        
        # Calculate directions
        directions = {k: self._direction_vector(self.history[k]) for k in self.history}
        
        # Calculate extensions
        arm_ext_l = self._extension(pos['left_shoulder'], pos['left_elbow'], pos['left_wrist'])
        arm_ext_r = self._extension(pos['right_shoulder'], pos['right_elbow'], pos['right_wrist'])
        leg_ext_l = self._extension(pos['left_hip'], pos['left_knee'], pos['left_ankle'])
        leg_ext_r = self._extension(pos['right_hip'], pos['right_knee'], pos['right_ankle'])
        
        result['extensions'] = {
            'left_arm': arm_ext_l,
            'right_arm': arm_ext_r,
            'left_leg': leg_ext_l,
            'right_leg': leg_ext_r,
        }
        
        # Store velocity data for analysis
        self.velocity_data['wrist'].append(max(vel['left_wrist'], vel['right_wrist']))
        self.velocity_data['ankle'].append(max(vel['left_ankle'], vel['right_ankle']))
        self.velocity_data['knee'].append(max(vel['left_knee'], vel['right_knee']))
        
        # Decrement cooldowns
        for k in self.cooldown:
            self.cooldown[k] = max(0, self.cooldown[k] - 1)
        
        # Helper: check if limb hits target
        def hits_target(pos_key: str) -> bool:
            p = pos.get(pos_key)
            if p is None or not target_boxes:
                return False
            return any(self._in_box(p, box, w, h) for box in target_boxes)
        
        # PUNCH DETECTION (Enhanced)
        for side, wrist_key, cd_key in [('left', 'left_wrist', 'punch_l'),
                                         ('right', 'right_wrist', 'punch_r')]:
            if self.cooldown[cd_key] > 0:
                continue
            
            wrist_vel = vel[wrist_key]
            wrist_accel = accel[wrist_key]
            wrist_dir = directions[wrist_key]
            wrist_pos = pos[wrist_key]
            arm_ext = arm_ext_l if side == 'left' else arm_ext_r
            
            # Multi-criteria check:
            # 1. Velocity in valid range
            # 2. Arm extension sufficient
            # 3. Moving towards target
            # 4. In target zone
            # 5. Decelerating (impact)
            
            if (self.punch_vel_min < wrist_vel < self.punch_vel_max and
                arm_ext > self.min_arm_ext and
                wrist_pos is not None and wrist_dir is not None and
                self._moving_towards_target(wrist_pos, wrist_dir, target_boxes, w, h) and
                hits_target(wrist_key) and
                wrist_accel < -self.punch_accel_thresh):  # Deceleration
                
                result['pose_punch_hit'] = True
                self.punch_count += 1
                self.cooldown[cd_key] = self.cooldown_frames
                
                # Record metrics
                self.strike_metrics.append({
                    'frame': frame_idx,
                    'type': 'punch',
                    'side': side,
                    'velocity': wrist_vel,
                    'acceleration': wrist_accel,
                    'extension': arm_ext,
                })
        
        # KICK DETECTION (Enhanced)
        for side, ankle_key, cd_key in [('left', 'left_ankle', 'kick_l'),
                                         ('right', 'right_ankle', 'kick_r')]:
            if self.cooldown[cd_key] > 0:
                continue
            
            ankle_vel = vel[ankle_key]
            ankle_accel = accel[ankle_key]
            ankle_dir = directions[ankle_key]
            ankle_pos = pos[ankle_key]
            leg_ext = leg_ext_l if side == 'left' else leg_ext_r
            
            if (self.kick_vel_min < ankle_vel < self.kick_vel_max and
                leg_ext > self.min_leg_ext and
                ankle_pos is not None and ankle_dir is not None and
                self._moving_towards_target(ankle_pos, ankle_dir, target_boxes, w, h) and
                hits_target(ankle_key) and
                ankle_accel < -self.kick_accel_thresh):
                
                result['pose_kick_hit'] = True
                self.kick_count += 1
                self.cooldown[cd_key] = self.cooldown_frames
                
                self.strike_metrics.append({
                    'frame': frame_idx,
                    'type': 'kick',
                    'side': side,
                    'velocity': ankle_vel,
                    'acceleration': ankle_accel,
                    'extension': leg_ext,
                })
        
        # KNEE DETECTION (Enhanced)
        for side, knee_key, cd_key in [('left', 'left_knee', 'knee_l'),
                                        ('right', 'right_knee', 'knee_r')]:
            if self.cooldown[cd_key] > 0:
                continue
            
            knee_vel = vel[knee_key]
            knee_accel = accel[knee_key]
            knee_dir = directions[knee_key]
            knee_pos = pos[knee_key]
            
            # Knees: lower extension requirement, velocity threshold
            if (self.kick_vel_min * 0.7 < knee_vel < self.kick_vel_max and
                knee_pos is not None and knee_dir is not None and
                self._moving_towards_target(knee_pos, knee_dir, target_boxes, w, h) and
                hits_target(knee_key) and
                knee_accel < -self.kick_accel_thresh * 0.8):
                
                result['pose_knee_hit'] = True
                self.knee_count += 1
                self.cooldown[cd_key] = self.cooldown_frames
                
                self.strike_metrics.append({
                    'frame': frame_idx,
                    'type': 'knee',
                    'side': side,
                    'velocity': knee_vel,
                    'acceleration': knee_accel,
                    'extension': 0,  # N/A for knees
                })
        
        # Update counts
        result['punch_count'] = self.punch_count
        result['kick_count'] = self.kick_count
        result['knee_count'] = self.knee_count
        
        return result
    
    def draw_overlay(self, frame: np.ndarray, result: Dict,
                     draw_skeleton: bool = True,
                     draw_velocity: bool = True) -> np.ndarray:
        """
        Draw enhanced visualization with velocity indicators.
        """
        landmarks = result.get('landmarks')
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Convert landmarks to pixel coordinates
        def to_pixel(lm):
            return (int(lm.x * w), int(lm.y * h))
        
        pts = [to_pixel(lm) for lm in landmarks]
        
        # Draw skeleton
        if draw_skeleton:
            for (a, b) in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        
        # Draw velocity indicators
        if draw_velocity:
            velocities = result.get('velocities', {})
            pos = result.get('positions', {})
            
            # Wrists (punches)
            for wrist_key, color in [('left_wrist', (255, 100, 255)),
                                      ('right_wrist', (255, 150, 255))]:
                p = pos.get(wrist_key)
                vel = velocities.get(wrist_key, 0)
                if p is not None:
                    px, py = int(p[0] * w), int(p[1] * h)
                    
                    # Velocity-based circle size and color intensity
                    vel_normalized = min(vel / self.punch_vel_max, 1.0)
                    radius = int(8 + vel_normalized * 20)
                    intensity = int(100 + vel_normalized * 155)
                    vel_color = (intensity, intensity//2, 255)
                    
                    cv2.circle(frame, (px, py), radius, vel_color, 2)
                    
                    # Velocity text
                    cv2.putText(frame, f"{vel:.2f}", (px+15, py-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, vel_color, 1)
            
            # Ankles (kicks)
            for ankle_key in ['left_ankle', 'right_ankle']:
                p = pos.get(ankle_key)
                vel = velocities.get(ankle_key, 0)
                if p is not None:
                    px, py = int(p[0] * w), int(p[1] * h)
                    
                    vel_normalized = min(vel / self.kick_vel_max, 1.0)
                    radius = int(8 + vel_normalized * 20)
                    intensity = int(100 + vel_normalized * 155)
                    vel_color = (100, intensity, intensity)
                    
                    cv2.circle(frame, (px, py), radius, vel_color, 2)
                    cv2.putText(frame, f"{vel:.2f}", (px+15, py-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, vel_color, 1)
        
        # Strike hit indicators
        if result.get('pose_punch_hit'):
            cv2.putText(frame, "PUNCH!", (50, h-100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if result.get('pose_kick_hit'):
            cv2.putText(frame, "KICK!", (50, h-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        if result.get('pose_knee_hit'):
            cv2.putText(frame, "KNEE!", (50, h-150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return frame
    
    def get_scientific_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive scientific metrics for analysis.
        
        Returns:
            Dictionary with statistical summaries
        """
        import pandas as pd
        
        metrics = {
            'total_strikes': len(self.strike_metrics),
            'punches': self.punch_count,
            'kicks': self.kick_count,
            'knees': self.knee_count,
        }
        
        if self.strike_metrics:
            df = pd.DataFrame(self.strike_metrics)
            
            # Per-type statistics
            for strike_type in ['punch', 'kick', 'knee']:
                subset = df[df['type'] == strike_type]
                if len(subset) > 0:
                    metrics[f'{strike_type}_velocity_mean'] = subset['velocity'].mean()
                    metrics[f'{strike_type}_velocity_std'] = subset['velocity'].std()
                    metrics[f'{strike_type}_velocity_max'] = subset['velocity'].max()
        
        # Velocity distributions
        metrics['velocity_data'] = self.velocity_data
        
        return metrics
