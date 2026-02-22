# modules/body_analyzer.py (COMPLETE FILE - Tasks API)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from config import POSE_LANDMARKER_MODEL


class PoseLandmarkIndex:
    """Pose landmark indices matching MediaPipe's 33-point model."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = [
    (PoseLandmarkIndex.LEFT_SHOULDER, PoseLandmarkIndex.RIGHT_SHOULDER),
    (PoseLandmarkIndex.LEFT_SHOULDER, PoseLandmarkIndex.LEFT_ELBOW),
    (PoseLandmarkIndex.LEFT_ELBOW, PoseLandmarkIndex.LEFT_WRIST),
    (PoseLandmarkIndex.RIGHT_SHOULDER, PoseLandmarkIndex.RIGHT_ELBOW),
    (PoseLandmarkIndex.RIGHT_ELBOW, PoseLandmarkIndex.RIGHT_WRIST),
    (PoseLandmarkIndex.LEFT_SHOULDER, PoseLandmarkIndex.LEFT_HIP),
    (PoseLandmarkIndex.RIGHT_SHOULDER, PoseLandmarkIndex.RIGHT_HIP),
    (PoseLandmarkIndex.LEFT_HIP, PoseLandmarkIndex.RIGHT_HIP),
    (PoseLandmarkIndex.LEFT_HIP, PoseLandmarkIndex.LEFT_KNEE),
    (PoseLandmarkIndex.LEFT_KNEE, PoseLandmarkIndex.LEFT_ANKLE),
    (PoseLandmarkIndex.RIGHT_HIP, PoseLandmarkIndex.RIGHT_KNEE),
    (PoseLandmarkIndex.RIGHT_KNEE, PoseLandmarkIndex.RIGHT_ANKLE),
    (PoseLandmarkIndex.LEFT_ANKLE, PoseLandmarkIndex.LEFT_HEEL),
    (PoseLandmarkIndex.RIGHT_ANKLE, PoseLandmarkIndex.RIGHT_HEEL),
    (PoseLandmarkIndex.LEFT_HEEL, PoseLandmarkIndex.LEFT_FOOT_INDEX),
    (PoseLandmarkIndex.RIGHT_HEEL, PoseLandmarkIndex.RIGHT_FOOT_INDEX),
    (PoseLandmarkIndex.LEFT_WRIST, PoseLandmarkIndex.LEFT_PINKY),
    (PoseLandmarkIndex.LEFT_WRIST, PoseLandmarkIndex.LEFT_INDEX),
    (PoseLandmarkIndex.LEFT_WRIST, PoseLandmarkIndex.LEFT_THUMB),
    (PoseLandmarkIndex.RIGHT_WRIST, PoseLandmarkIndex.RIGHT_PINKY),
    (PoseLandmarkIndex.RIGHT_WRIST, PoseLandmarkIndex.RIGHT_INDEX),
    (PoseLandmarkIndex.RIGHT_WRIST, PoseLandmarkIndex.RIGHT_THUMB),
    (PoseLandmarkIndex.NOSE, PoseLandmarkIndex.LEFT_EYE),
    (PoseLandmarkIndex.NOSE, PoseLandmarkIndex.RIGHT_EYE),
    (PoseLandmarkIndex.LEFT_EYE, PoseLandmarkIndex.LEFT_EAR),
    (PoseLandmarkIndex.RIGHT_EYE, PoseLandmarkIndex.RIGHT_EAR),
]


@dataclass
class BodyMovementData:
    """Body movement analysis results."""
    pose_detected: bool = False
    overall_movement_score: float = 0.0
    is_rocking: bool = False
    rocking_frequency: float = 0.0
    is_hand_flapping: bool = False
    hand_flap_score: float = 0.0
    is_toe_walking: bool = False
    body_symmetry_score: float = 0.0
    stillness_score: float = 0.0
    repetitive_motion_score: float = 0.0
    landmarks_3d: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None


class BodyAnalyzer:
    """
    Analyze body movements using MediaPipe Tasks API
    PoseLandmarker.

    Detects autism-related motor behaviors:
    - Body rocking (repetitive torso oscillation)
    - Hand flapping (rapid hand movements)
    - Unusual stillness
    - Repetitive movements (autocorrelation)
    """

    def __init__(self):
        base_options = mp_python.BaseOptions(
            model_asset_path=POSE_LANDMARKER_MODEL
        )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )

        self.detector = mp_vision.PoseLandmarker.create_from_options(
            options
        )
        print("✅ PoseLandmarker (Tasks API) initialized")

        self.LM = PoseLandmarkIndex

        # Movement history buffers
        self.HISTORY_SIZE = 90
        self.torso_y_history = deque(maxlen=self.HISTORY_SIZE)
        self.left_wrist_history = deque(maxlen=self.HISTORY_SIZE)
        self.right_wrist_history = deque(maxlen=self.HISTORY_SIZE)
        self.shoulder_center_history = deque(maxlen=self.HISTORY_SIZE)

        # Movement events
        self.movement_events: List[dict] = []
        self.start_time = time.time()

        # Detection counters
        self.rocking_detected_frames = 0
        self.rocking_total_frames = 0
        self.flap_detected_frames = 0

    def _get_landmark_xyz(
        self, landmarks, index: int
    ) -> np.ndarray:
        """Extract x, y, z from a landmark by index."""
        lm = landmarks[index]
        return np.array([lm.x, lm.y, lm.z])

    def _get_landmark_visibility(
        self, landmarks, index: int
    ) -> float:
        """Get visibility/presence score of a landmark."""
        lm = landmarks[index]
        return getattr(
            lm, 'visibility',
            getattr(lm, 'presence', 0.5)
        )

    def _compute_rocking(self) -> Tuple[bool, float]:
        """
        Detect body rocking by analyzing periodic
        vertical oscillation of the torso center.
        Uses zero-crossing rate of detrended signal.
        """
        if len(self.torso_y_history) < 30:
            return False, 0.0

        signal = np.array(self.torso_y_history)
        trend = np.convolve(
            signal, np.ones(15) / 15, mode='same'
        )
        detrended = signal - trend

        zero_crossings = np.sum(
            np.abs(np.diff(np.sign(detrended))) > 0
        )
        frequency = zero_crossings / (len(signal) / 30.0)
        amplitude = np.std(detrended)

        is_rocking = (
            0.5 < frequency < 3.0 and amplitude > 0.008
        )
        return is_rocking, round(frequency, 2)

    def _compute_hand_flapping(self) -> Tuple[bool, float]:
        """
        Detect hand flapping by measuring rapid
        oscillatory wrist movements.
        """
        if len(self.left_wrist_history) < 20:
            return False, 0.0

        left_signal = np.array(self.left_wrist_history)
        right_signal = np.array(self.right_wrist_history)

        left_velocity = np.diff(left_signal, axis=0)
        right_velocity = np.diff(right_signal, axis=0)

        left_speed = np.linalg.norm(left_velocity, axis=1)
        right_speed = np.linalg.norm(right_velocity, axis=1)

        left_reversals = np.sum(
            np.abs(np.diff(np.sign(left_velocity[:, 1]))) > 0
        )
        right_reversals = np.sum(
            np.abs(np.diff(np.sign(right_velocity[:, 1]))) > 0
        )

        avg_speed = (
            np.mean(left_speed) + np.mean(right_speed)
        ) / 2
        avg_reversals = (left_reversals + right_reversals) / 2
        reversal_rate = avg_reversals / (len(left_speed) / 30.0)

        flap_score = min(avg_speed * reversal_rate * 10, 1.0)
        is_flapping = flap_score > 0.4

        return is_flapping, round(flap_score, 3)

    def _compute_repetitive_motion(self) -> float:
        """
        Detect general repetitive motion using
        autocorrelation of the shoulder center trajectory.
        """
        if len(self.shoulder_center_history) < 60:
            return 0.0

        signal = np.array(self.shoulder_center_history)
        y_signal = signal[:, 1]

        y_norm = y_signal - np.mean(y_signal)
        if np.std(y_norm) < 1e-6:
            return 0.0
        y_norm = y_norm / np.std(y_norm)

        n = len(y_norm)
        autocorr = np.correlate(y_norm, y_norm, mode='full')
        autocorr = autocorr[n - 1:]
        autocorr = autocorr / autocorr[0]

        search_region = autocorr[15:min(90, len(autocorr))]
        if len(search_region) == 0:
            return 0.0

        max_autocorr = np.max(search_region)
        return float(max(0, min(max_autocorr, 1.0)))

    def _draw_pose_landmarks(
        self,
        frame: np.ndarray,
        landmarks,
        img_w: int,
        img_h: int,
        is_rocking: bool = False,
        is_flapping: bool = False
    ) -> np.ndarray:
        """
        Draw pose landmarks and connections manually
        on the frame with color-coded alerts.
        """
        annotated = frame.copy()

        # Draw connections (bones)
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_vis = self._get_landmark_visibility(
                landmarks, start_idx
            )
            end_vis = self._get_landmark_visibility(
                landmarks, end_idx
            )

            if start_vis > 0.3 and end_vis > 0.3:
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]

                x1 = int(start_lm.x * img_w)
                y1 = int(start_lm.y * img_h)
                x2 = int(end_lm.x * img_w)
                y2 = int(end_lm.y * img_h)

                # Color bones based on alerts
                torso_joints = {
                    self.LM.LEFT_SHOULDER,
                    self.LM.RIGHT_SHOULDER,
                    self.LM.LEFT_HIP,
                    self.LM.RIGHT_HIP
                }
                wrist_joints = {
                    self.LM.LEFT_WRIST,
                    self.LM.RIGHT_WRIST
                }

                if (
                    is_rocking and
                    start_idx in torso_joints and
                    end_idx in torso_joints
                ):
                    bone_color = (0, 0, 255)  # Red for rocking
                    thickness = 3
                elif (
                    is_flapping and (
                        start_idx in wrist_joints or
                        end_idx in wrist_joints
                    )
                ):
                    bone_color = (0, 165, 255)  # Orange for flapping
                    thickness = 3
                else:
                    bone_color = (0, 255, 0)  # Green normal
                    thickness = 2

                cv2.line(
                    annotated,
                    (x1, y1), (x2, y2),
                    bone_color, thickness
                )

        # Draw joint points
        for i in range(len(landmarks)):
            vis = self._get_landmark_visibility(landmarks, i)
            if vis > 0.3:
                lm = landmarks[i]
                x = int(lm.x * img_w)
                y = int(lm.y * img_h)

                # Color code key joints with alerts
                if i in [self.LM.LEFT_WRIST, self.LM.RIGHT_WRIST]:
                    if is_flapping:
                        color = (0, 0, 255)   # Red alert
                        radius = 8
                    else:
                        color = (0, 165, 255)  # Orange
                        radius = 6
                elif i in [
                    self.LM.LEFT_SHOULDER,
                    self.LM.RIGHT_SHOULDER,
                    self.LM.LEFT_HIP,
                    self.LM.RIGHT_HIP
                ]:
                    if is_rocking:
                        color = (0, 0, 255)   # Red alert
                        radius = 7
                    else:
                        color = (255, 0, 0)    # Blue
                        radius = 5
                elif i in [
                    self.LM.LEFT_ANKLE,
                    self.LM.RIGHT_ANKLE
                ]:
                    color = (0, 255, 255)  # Yellow
                    radius = 5
                elif i == self.LM.NOSE:
                    color = (255, 255, 255)  # White
                    radius = 5
                else:
                    color = (0, 200, 0)    # Green
                    radius = 3

                cv2.circle(annotated, (x, y), radius, color, -1)
                # White border for visibility
                cv2.circle(
                    annotated, (x, y), radius,
                    (255, 255, 255), 1
                )

        # Draw alert banners at top if detected
        banner_y = 0
        if is_rocking:
            cv2.rectangle(
                annotated,
                (0, banner_y),
                (img_w, banner_y + 30),
                (0, 0, 180), -1
            )
            cv2.putText(
                annotated,
                "⚠ BODY ROCKING DETECTED",
                (10, banner_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )
            banner_y += 32

        if is_flapping:
            cv2.rectangle(
                annotated,
                (0, banner_y),
                (img_w, banner_y + 30),
                (0, 100, 180), -1
            )
            cv2.putText(
                annotated,
                "⚠ HAND FLAPPING DETECTED",
                (10, banner_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

        return annotated

    def analyze_frame(
        self, frame: np.ndarray
    ) -> BodyMovementData:
        """Analyze a single frame for body movements."""
        result = BodyMovementData()
        img_h, img_w = frame.shape[:2]

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect pose
        detection_result = self.detector.detect(mp_image)

        annotated = frame.copy()

        # Check if pose detected
        if (
            not detection_result.pose_landmarks
            or len(detection_result.pose_landmarks) == 0
        ):
            result.pose_detected = False
            cv2.putText(
                annotated, "NO POSE DETECTED",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2
            )
            result.annotated_frame = annotated
            return result

        result.pose_detected = True
        landmarks = detection_result.pose_landmarks[0]

        # Store 3D landmarks as numpy array (33 x 4)
        result.landmarks_3d = np.array([
            (
                lm.x, lm.y, lm.z,
                getattr(
                    lm, 'visibility',
                    getattr(lm, 'presence', 0.5)
                )
            )
            for lm in landmarks
        ])

        # Extract key body points
        left_shoulder = self._get_landmark_xyz(
            landmarks, self.LM.LEFT_SHOULDER
        )
        right_shoulder = self._get_landmark_xyz(
            landmarks, self.LM.RIGHT_SHOULDER
        )
        left_hip = self._get_landmark_xyz(
            landmarks, self.LM.LEFT_HIP
        )
        right_hip = self._get_landmark_xyz(
            landmarks, self.LM.RIGHT_HIP
        )
        left_wrist = self._get_landmark_xyz(
            landmarks, self.LM.LEFT_WRIST
        )
        right_wrist = self._get_landmark_xyz(
            landmarks, self.LM.RIGHT_WRIST
        )

        # Torso and shoulder centers
        torso_center = (
            left_shoulder + right_shoulder +
            left_hip + right_hip
        ) / 4
        shoulder_center = (
            left_shoulder + right_shoulder
        ) / 2

        # Update histories
        self.torso_y_history.append(torso_center[1])
        self.left_wrist_history.append(left_wrist)
        self.right_wrist_history.append(right_wrist)
        self.shoulder_center_history.append(shoulder_center)

        # --- Rocking Detection ---
        is_rocking, rock_freq = self._compute_rocking()
        result.is_rocking = is_rocking
        result.rocking_frequency = rock_freq
        if is_rocking:
            self.rocking_detected_frames += 1
        self.rocking_total_frames += 1

        # --- Hand Flapping ---
        is_flapping, flap_score = self._compute_hand_flapping()
        result.is_hand_flapping = is_flapping
        result.hand_flap_score = flap_score
        if is_flapping:
            self.flap_detected_frames += 1

        # --- Repetitive Motion ---
        result.repetitive_motion_score = (
            self._compute_repetitive_motion()
        )

        # --- Stillness ---
        if len(self.shoulder_center_history) > 10:
            recent = np.array(
                list(self.shoulder_center_history)[-10:]
            )
            result.stillness_score = float(
                1.0 - min(np.std(recent) * 100, 1.0)
            )

        # --- Overall Movement ---
        if len(self.shoulder_center_history) > 2:
            positions = np.array(
                list(self.shoulder_center_history)
            )
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            result.overall_movement_score = float(
                np.mean(speeds)
            )

        # --- Draw Pose with Alert Colors ---
        annotated = self._draw_pose_landmarks(
            frame, landmarks, img_w, img_h,
            is_rocking=is_rocking,
            is_flapping=is_flapping
        )

        # Overlay stats text (below any alert banners)
        text_y_start = 30
        if is_rocking:
            text_y_start += 32
        if is_flapping:
            text_y_start += 32

        rock_color = (0, 0, 255) if is_rocking else (0, 255, 0)
        cv2.putText(
            annotated,
            f"Rocking: {'YES' if is_rocking else 'No'} "
            f"({rock_freq:.1f} Hz)",
            (10, text_y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, rock_color, 2
        )

        flap_color = (0, 0, 255) if is_flapping else (0, 255, 0)
        cv2.putText(
            annotated,
            f"Flapping: {'YES' if is_flapping else 'No'} "
            f"({flap_score:.2f})",
            (10, text_y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, flap_color, 2
        )

        cv2.putText(
            annotated,
            f"Repetitive: {result.repetitive_motion_score:.2f}",
            (10, text_y_start + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 200, 0), 1
        )

        cv2.putText(
            annotated,
            f"Stillness: {result.stillness_score:.2f} | "
            f"Movement: {result.overall_movement_score:.4f}",
            (10, text_y_start + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1
        )

        result.annotated_frame = annotated
        return result

    def get_session_stats(self) -> dict:
        """Get accumulated body movement statistics."""
        elapsed = time.time() - self.start_time
        rock_pct = (
            (self.rocking_detected_frames /
             max(self.rocking_total_frames, 1)) * 100
        )
        flap_pct = (
            (self.flap_detected_frames /
             max(self.rocking_total_frames, 1)) * 100
        )

        return {
            "rocking_percentage": round(rock_pct, 1),
            "flapping_percentage": round(flap_pct, 1),
            "rocking_detected_frames": self.rocking_detected_frames,
            "flapping_detected_frames": self.flap_detected_frames,
            "total_frames_analyzed": self.rocking_total_frames,
            "session_duration": round(elapsed, 1)
        }

    def reset(self):
        """Reset for new session."""
        self.torso_y_history.clear()
        self.left_wrist_history.clear()
        self.right_wrist_history.clear()
        self.shoulder_center_history.clear()
        self.movement_events = []
        self.rocking_detected_frames = 0
        self.rocking_total_frames = 0
        self.flap_detected_frames = 0
        self.start_time = time.time()

    def close(self):
        """Clean up the detector."""
        if self.detector:
            self.detector.close()