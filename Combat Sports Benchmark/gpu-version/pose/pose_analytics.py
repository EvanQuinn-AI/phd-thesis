# pose_analytics.py
# MediaPipe Pose Analytics for Combat Sports Training
# Uses mp.solutions.pose (no .task file download needed)
# Provides velocity, acceleration, extension, and biomechanical metrics
# Strike COUNTING is handled by YOLO - this module provides the ANALYTICS layer

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import List, Tuple, Optional, Dict, Any


# =============================================================================
# LANDMARK INDICES (MediaPipe Pose)
# =============================================================================

LANDMARKS = {
    'nose': 0,
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

# Tracked limb endpoints for velocity/acceleration
TRACKED_JOINTS = [
    'left_wrist', 'right_wrist',
    'left_ankle', 'right_ankle',
    'left_knee', 'right_knee',
    'left_elbow', 'right_elbow',
]


# =============================================================================
# POSE ANALYTICS ENGINE
# =============================================================================

class PoseAnalytics:
    """
    Biomechanical analytics from MediaPipe Pose for combat sports training.
    
    This module does NOT count strikes — YOLO handles that.
    Instead it provides:
    - Joint velocities (how fast strikes are thrown)
    - Accelerations (snap/power of strikes) 
    - Limb extensions (technique quality — full extension vs short)
    - Guard position tracking (hands up/down)
    - Strike power estimation (velocity at moment of YOLO-detected impact)
    - Per-strike analytics correlated with YOLO detections
    """

    def __init__(self,
                 history_length: int = 7,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            history_length: Frames of position history for velocity calc
            model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy)
            min_detection_confidence: Pose detection threshold
            min_tracking_confidence: Pose tracking threshold
        """
        self.history_length = history_length
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # MediaPipe Pose (simple API — no .task file needed)
        self.mp_pose = mp.solutions.pose
        self.pose = None

        # Position histories for each tracked joint
        self.position_history: Dict[str, deque] = {
            joint: deque(maxlen=history_length) for joint in TRACKED_JOINTS
        }

        # Velocity histories for acceleration calculation
        self.velocity_history: Dict[str, deque] = {
            joint: deque(maxlen=5) for joint in TRACKED_JOINTS
        }

        # Collected metrics across entire video
        self.all_velocities: Dict[str, List[float]] = {joint: [] for joint in TRACKED_JOINTS}
        self.strike_analytics: List[Dict] = []  # Per-strike metrics (correlated with YOLO)
        self.guard_data: List[Dict] = []
        self.frame_count = 0

    def initialize(self) -> None:
        """Initialize MediaPipe Pose"""
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def close(self) -> None:
        """Release resources"""
        if self.pose:
            self.pose.close()
            self.pose = None

    def reset(self) -> None:
        """Reset all state"""
        for h in self.position_history.values():
            h.clear()
        for h in self.velocity_history.values():
            h.clear()
        self.all_velocities = {joint: [] for joint in TRACKED_JOINTS}
        self.strike_analytics = []
        self.guard_data = []
        self.frame_count = 0

    # =========================================================================
    # CORE COMPUTATION
    # =========================================================================

    def _get_pos(self, landmarks, name: str, vis_thresh: float = 0.5) -> Optional[Tuple[float, float]]:
        """Get normalized (0-1) position of a named landmark"""
        idx = LANDMARKS.get(name)
        if idx is None or landmarks is None:
            return None
        lm = landmarks.landmark[idx]
        if lm.visibility < vis_thresh:
            return None
        return (lm.x, lm.y)

    def _velocity(self, history: deque) -> float:
        """Instantaneous velocity from recent position history (normalized units/frame)"""
        positions = [p for p in history if p is not None]
        if len(positions) < 2:
            return 0.0
        # Use last 3 frames for responsive velocity
        recent = positions[-3:] if len(positions) >= 3 else positions[-2:]
        total = sum(
            np.sqrt((recent[i][0] - recent[i - 1][0]) ** 2 +
                    (recent[i][1] - recent[i - 1][1]) ** 2)
            for i in range(1, len(recent))
        )
        return total / (len(recent) - 1)

    def _acceleration(self, vel_history: deque) -> float:
        """Acceleration (change in velocity). Negative = deceleration (impact)."""
        if len(vel_history) < 2:
            return 0.0
        vals = list(vel_history)
        return vals[-1] - vals[-2]

    def _extension(self, p1, p2, p3) -> float:
        """
        Limb extension ratio: 0 = fully bent, 1 = fully straight.
        p1=proximal joint, p2=middle joint, p3=distal joint.
        """
        if any(p is None for p in [p1, p2, p3]):
            return 0.0
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        angle = np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1, 1))
        return angle / np.pi  # 0=folded, 1=straight

    def _guard_height(self, wrist_pos, shoulder_pos, hip_pos) -> Optional[float]:
        """
        Guard height metric: 1.0 = hands at shoulder level, 0.0 = hands at hip level.
        > 0.7 is high guard, < 0.3 is low guard.
        """
        if any(p is None for p in [wrist_pos, shoulder_pos, hip_pos]):
            return None
        torso_height = abs(hip_pos[1] - shoulder_pos[1])
        if torso_height < 1e-6:
            return None
        # How far up the wrist is between hip and shoulder (inverted because y increases downward)
        ratio = (hip_pos[1] - wrist_pos[1]) / torso_height
        return max(0.0, min(1.0, ratio))

    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================

    def process_frame(self, frame: np.ndarray, frame_idx: int,
                      yolo_punch_hit: bool = False,
                      yolo_kick_hit: bool = False) -> Dict[str, Any]:
        """
        Process a single frame for pose analytics.
        
        Args:
            frame: BGR frame from OpenCV
            frame_idx: Frame number
            yolo_punch_hit: Whether YOLO detected a punch hit this frame
            yolo_kick_hit: Whether YOLO detected a kick hit this frame
            
        Returns:
            Dictionary with all computed metrics for this frame
        """
        h, w = frame.shape[:2]
        self.frame_count += 1

        result = {
            'frame': frame_idx,
            'pose_detected': False,
            'landmarks': None,
            'positions': {},
            'velocities': {},
            'accelerations': {},
            'extensions': {},
            'guard': {},
        }

        # Run pose detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = self.pose.process(rgb)

        if not detection.pose_landmarks:
            # Still update histories with None so velocity doesn't stall
            for joint in TRACKED_JOINTS:
                self.position_history[joint].append(None)
            return result

        result['pose_detected'] = True
        landmarks = detection.pose_landmarks
        result['landmarks'] = landmarks

        # Extract positions
        pos: Dict[str, Optional[Tuple[float, float]]] = {}
        for name in LANDMARKS:
            pos[name] = self._get_pos(landmarks, name)
        result['positions'] = pos

        # Update position histories & compute velocities
        vel: Dict[str, float] = {}
        accel: Dict[str, float] = {}
        for joint in TRACKED_JOINTS:
            self.position_history[joint].append(pos.get(joint))
            v = self._velocity(self.position_history[joint])
            vel[joint] = v
            self.velocity_history[joint].append(v)
            accel[joint] = self._acceleration(self.velocity_history[joint])
            self.all_velocities[joint].append(v)

        result['velocities'] = vel
        result['accelerations'] = accel

        # Compute limb extensions
        extensions = {
            'left_arm': self._extension(pos.get('left_shoulder'), pos.get('left_elbow'), pos.get('left_wrist')),
            'right_arm': self._extension(pos.get('right_shoulder'), pos.get('right_elbow'), pos.get('right_wrist')),
            'left_leg': self._extension(pos.get('left_hip'), pos.get('left_knee'), pos.get('left_ankle')),
            'right_leg': self._extension(pos.get('right_hip'), pos.get('right_knee'), pos.get('right_ankle')),
        }
        result['extensions'] = extensions

        # Guard position
        guard = {}
        for side in ['left', 'right']:
            guard[f'{side}_guard'] = self._guard_height(
                pos.get(f'{side}_wrist'),
                pos.get(f'{side}_shoulder'),
                pos.get(f'{side}_hip')
            )
        result['guard'] = guard

        # Self-contained guard data collection
        self.guard_data.append({
            'frame': frame_idx,
            'left_guard': guard.get('left_guard'),
            'right_guard': guard.get('right_guard'),
        })

        # If YOLO detected a strike this frame, record the biomechanics at impact
        if yolo_punch_hit:
            max_wrist_vel = max(vel.get('left_wrist', 0), vel.get('right_wrist', 0))
            # Figure out which hand
            if vel.get('left_wrist', 0) >= vel.get('right_wrist', 0):
                striking_side = 'left'
            else:
                striking_side = 'right'
            self.strike_analytics.append({
                'frame': frame_idx,
                'type': 'punch',
                'side': striking_side,
                'velocity': max_wrist_vel,
                'acceleration': accel.get(f'{striking_side}_wrist', 0),
                'arm_extension': extensions.get(f'{striking_side}_arm', 0),
                'guard_height': guard.get(f'{striking_side}_guard'),
            })

        if yolo_kick_hit:
            max_ankle_vel = max(vel.get('left_ankle', 0), vel.get('right_ankle', 0))
            max_knee_vel = max(vel.get('left_knee', 0), vel.get('right_knee', 0))
            if vel.get('left_ankle', 0) >= vel.get('right_ankle', 0):
                striking_side = 'left'
            else:
                striking_side = 'right'
            self.strike_analytics.append({
                'frame': frame_idx,
                'type': 'kick',
                'side': striking_side,
                'velocity': max_ankle_vel,
                'knee_velocity': max_knee_vel,
                'acceleration': accel.get(f'{striking_side}_ankle', 0),
                'leg_extension': extensions.get(f'{striking_side}_leg', 0),
                'guard_height': guard.get(f'{striking_side}_guard'),
            })

        return result

    # =========================================================================
    # OVERLAY DRAWING
    # =========================================================================

    def draw_overlay(self, frame: np.ndarray, result: Dict,
                     draw_skeleton: bool = True,
                     draw_velocity: bool = True) -> np.ndarray:
        """Draw pose skeleton and velocity indicators on frame."""
        landmarks = result.get('landmarks')
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # Convert to pixel coordinates
        pts = {}
        for name, idx in LANDMARKS.items():
            lm = landmarks.landmark[idx]
            if lm.visibility > 0.5:
                pts[idx] = (int(lm.x * w), int(lm.y * h))

        # Draw skeleton
        if draw_skeleton:
            for (a, b) in POSE_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for idx, pt in pts.items():
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        # Draw velocity indicators on wrists and ankles
        if draw_velocity:
            vel = result.get('velocities', {})
            positions = result.get('positions', {})

            for joint, color_base in [
                ('left_wrist', (255, 100, 255)),
                ('right_wrist', (255, 150, 255)),
                ('left_ankle', (100, 255, 255)),
                ('right_ankle', (150, 255, 255)),
            ]:
                p = positions.get(joint)
                v = vel.get(joint, 0)
                if p is not None:
                    px, py = int(p[0] * w), int(p[1] * h)
                    # Scale circle by velocity
                    v_norm = min(v / 0.15, 1.0)
                    radius = int(8 + v_norm * 20)
                    intensity = int(100 + v_norm * 155)
                    if 'wrist' in joint:
                        c = (intensity, intensity // 2, 255)
                    else:
                        c = (100, intensity, intensity)
                    cv2.circle(frame, (px, py), radius, c, 2)
                    cv2.putText(frame, f"{v:.3f}", (px + 15, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        # Draw guard indicators
        guard = result.get('guard', {})
        for side, x_offset in [('left', 10), ('right', w - 120)]:
            g = guard.get(f'{side}_guard')
            if g is not None:
                bar_h = int(g * 60)
                color = (0, 255, 0) if g > 0.6 else (0, 165, 255) if g > 0.3 else (0, 0, 255)
                cv2.rectangle(frame, (x_offset, h - 80), (x_offset + 20, h - 80 + 60), (50, 50, 50), -1)
                cv2.rectangle(frame, (x_offset, h - 80 + (60 - bar_h)), (x_offset + 20, h - 80 + 60), color, -1)
                cv2.putText(frame, f"{side[0].upper()} Guard", (x_offset, h - 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return frame

    # =========================================================================
    # AGGREGATE METRICS
    # =========================================================================

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics for the entire processed video.
        
        Returns dict with:
        - Per-joint velocity statistics
        - Per-strike-type biomechanics (velocity, extension at impact)
        - Guard analysis
        """
        metrics: Dict[str, Any] = {
            'frames_processed': self.frame_count,
            'strikes_analyzed': len(self.strike_analytics),
        }

        # Joint velocity statistics
        for joint in ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle',
                       'left_knee', 'right_knee']:
            data = self.all_velocities.get(joint, [])
            if data:
                arr = np.array(data)
                metrics[f'{joint}_vel_mean'] = float(np.mean(arr))
                metrics[f'{joint}_vel_std'] = float(np.std(arr))
                metrics[f'{joint}_vel_max'] = float(np.max(arr))
                metrics[f'{joint}_vel_p95'] = float(np.percentile(arr, 95))

        # Strike-type biomechanics (from YOLO-correlated impacts)
        if self.strike_analytics:
            import pandas as pd
            df = pd.DataFrame(self.strike_analytics)
            for strike_type in ['punch', 'kick']:
                subset = df[df['type'] == strike_type]
                if len(subset) > 0:
                    metrics[f'{strike_type}_count'] = len(subset)
                    metrics[f'{strike_type}_impact_vel_mean'] = float(subset['velocity'].mean())
                    metrics[f'{strike_type}_impact_vel_max'] = float(subset['velocity'].max())
                    metrics[f'{strike_type}_impact_vel_std'] = float(subset['velocity'].std()) if len(subset) > 1 else 0.0
                    if 'arm_extension' in subset.columns:
                        ext_col = 'arm_extension' if strike_type == 'punch' else 'leg_extension'
                        if ext_col in subset.columns:
                            metrics[f'{strike_type}_ext_mean'] = float(subset[ext_col].mean())

        # Guard analysis
        if self.guard_data:
            import pandas as pd
            gdf = pd.DataFrame(self.guard_data)
            for side in ['left', 'right']:
                col = f'{side}_guard'
                valid = gdf[col].dropna()
                if len(valid) > 0:
                    metrics[f'{side}_guard_mean'] = float(valid.mean())
                    metrics[f'{side}_guard_high_pct'] = float((valid > 0.6).mean() * 100)

        return metrics
