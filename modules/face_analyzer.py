# modules/face_analyzer.py (COMPLETE REWRITE - Tasks API)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.spatial import distance
from config import FACE_LANDMARKER_MODEL


@dataclass
class GazeData:
    """Gaze analysis results for a single frame."""
    is_looking_at_camera: bool = False
    gaze_direction: str = "unknown"
    left_ear: float = 0.0
    right_ear: float = 0.0
    is_blinking: bool = False
    head_pose_pitch: float = 0.0
    head_pose_yaw: float = 0.0
    head_pose_roll: float = 0.0


@dataclass
class EmotionData:
    """Emotion estimation from face blendshapes + landmarks."""
    mouth_openness: float = 0.0
    brow_raise: float = 0.0
    smile_score: float = 0.0
    expression_label: str = "neutral"
    expression_variance: float = 0.0
    # NEW: Direct blendshape values from Tasks API
    eye_blink_left: float = 0.0
    eye_blink_right: float = 0.0
    jaw_open: float = 0.0
    mouth_smile_left: float = 0.0
    mouth_smile_right: float = 0.0
    brow_up_left: float = 0.0
    brow_up_right: float = 0.0


@dataclass
class FaceAnalysisResult:
    """Complete face analysis for one frame."""
    face_detected: bool = False
    gaze: GazeData = field(default_factory=GazeData)
    emotion: EmotionData = field(default_factory=EmotionData)
    landmarks_3d: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    timestamp: float = 0.0


class FaceAnalyzer:
    """
    Advanced face analysis using MediaPipe Tasks API
    FaceLandmarker.

    Extracts: Gaze direction, Eye contact, Blink rate,
    Head pose (via transformation matrix), Facial
    expressions (via blendshapes), Flat affect detection.
    """

    # Face Mesh landmark indices (same as before)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    NOSE_TIP = 1
    CHIN = 199
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # Blendshape indices (MediaPipe face blendshapes)
    BLENDSHAPE_NAMES = {
        'eyeBlinkLeft': None,
        'eyeBlinkRight': None,
        'jawOpen': None,
        'mouthSmileLeft': None,
        'mouthSmileRight': None,
        'browOuterUpLeft': None,
        'browOuterUpRight': None,
        'mouthFrownLeft': None,
        'mouthFrownRight': None,
        'eyeSquintLeft': None,
        'eyeSquintRight': None,
    }

    def __init__(self):
        # --- Create FaceLandmarker ---
        base_options = mp_python.BaseOptions(
            model_asset_path=FACE_LANDMARKER_MODEL
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        self.detector = mp_vision.FaceLandmarker.create_from_options(
            options
        )
        print("✅ FaceLandmarker (Tasks API) initialized")

        # Blink tracking
        self.blink_counter = 0
        self.blink_total = 0
        self.left_eye_closed_prev = False
        self.right_eye_closed_prev = False
        self.BLINK_THRESHOLD = 0.5  # Blendshape value

        # Gaze tracking
        self.gaze_away_start: Optional[float] = None
        self.total_gaze_away_time = 0.0
        self.gaze_events: List[dict] = []

        # Expression tracking
        self.expression_history: List[float] = []
        self.EXPRESSION_WINDOW = 90

        # Timing
        self.start_time = time.time()

    def _extract_blendshapes(self, blendshapes) -> dict:
        """
        Extract relevant blendshape values from
        FaceLandmarker result.
        """
        values = {}
        if blendshapes and len(blendshapes) > 0:
            for category in blendshapes[0]:
                if category.category_name in self.BLENDSHAPE_NAMES:
                    values[category.category_name] = category.score
        return values

    def _extract_head_pose(self, transformation_matrix) -> Tuple[float, float, float]:
        """
        Extract head pose (pitch, yaw, roll) from the
        facial transformation matrix provided by the
        Tasks API.
        """
        if (
            transformation_matrix is None
            or len(transformation_matrix) == 0
        ):
            return 0.0, 0.0, 0.0

        try:
            mat = np.array(transformation_matrix[0])  # 4x4 matrix
            # Extract rotation submatrix (3x3)
            rotation = mat[:3, :3]

            # Extract Euler angles
            sy = np.sqrt(
                rotation[0, 0] ** 2 + rotation[1, 0] ** 2
            )
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(
                    rotation[2, 1], rotation[2, 2]
                )
                yaw = np.arctan2(
                    -rotation[2, 0], sy
                )
                roll = np.arctan2(
                    rotation[1, 0], rotation[0, 0]
                )
            else:
                pitch = np.arctan2(
                    -rotation[1, 2], rotation[1, 1]
                )
                yaw = np.arctan2(
                    -rotation[2, 0], sy
                )
                roll = 0.0

            # Convert to degrees
            return (
                float(np.degrees(pitch)),
                float(np.degrees(yaw)),
                float(np.degrees(roll))
            )
        except Exception:
            return 0.0, 0.0, 0.0

    def _compute_ear(
        self,
        landmarks: np.ndarray,
        eye_indices: List[int]
    ) -> float:
        """
        Compute Eye Aspect Ratio (EAR).
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        pts = landmarks[eye_indices]
        vertical_1 = distance.euclidean(pts[1], pts[5])
        vertical_2 = distance.euclidean(pts[2], pts[4])
        horizontal = distance.euclidean(pts[0], pts[3])

        if horizontal == 0:
            return 0.0
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    # 1. Update this function in modules/face_analyzer.py
    def _compute_gaze_direction(
        self,
        landmarks: np.ndarray,
        img_w: int,
        img_h: int,
        head_yaw: float,   # <-- ADDED PARAMETER
        head_pitch: float  # <-- ADDED PARAMETER
    ) -> Tuple[str, bool]:
        """
        Determine gaze using Head Pose as the primary anchor, 
        falling back to iris tracking for micro-movements.
        """
        # 1. Primary Check: Head Pose (Bulletproof)
        # If head is turned significantly, eye contact is broken.
        if head_yaw > 18.0:
            return "right (head turned)", False
        elif head_yaw < -18.0:
            return "left (head turned)", False
        elif head_pitch > 20.0:
            return "down (head lowered)", False
        elif head_pitch < -20.0:
            return "up (head raised)", False

        # 2. Secondary Check: Iris Position
        if len(landmarks) <= max(max(self.LEFT_IRIS), max(self.RIGHT_IRIS)):
            return "center", True # Default to center if iris fails but head is straight

        left_iris_pts = landmarks[self.LEFT_IRIS]
        right_iris_pts = landmarks[self.RIGHT_IRIS]
        left_iris_center = left_iris_pts.mean(axis=0)
        right_iris_center = right_iris_pts.mean(axis=0)

        left_eye_pts = landmarks[self.LEFT_EYE]
        right_eye_pts = landmarks[self.RIGHT_EYE]

        def iris_ratio(iris_center, eye_pts):
            eye_left_corner = eye_pts[0]
            eye_right_corner = eye_pts[3]
            total_width = distance.euclidean(eye_left_corner[:2], eye_right_corner[:2])
            if total_width == 0: return 0.5
            iris_dist = distance.euclidean(eye_left_corner[:2], iris_center[:2])
            return iris_dist / total_width

        avg_ratio = (iris_ratio(left_iris_center, left_eye_pts) + iris_ratio(right_iris_center, right_eye_pts)) / 2

        # Widened thresholds to account for webcam noise
        if avg_ratio < 0.28:
            return "right (eyes)", False
        elif avg_ratio > 0.72:
            return "left (eyes)", False
        
        return "center", True

    def _analyze_expression_blendshapes(
        self,
        blendshapes: dict
    ) -> EmotionData:
        """
        Analyze facial expressions using blendshape values.
        This is MUCH more accurate than landmark distances!
        """
        emotion = EmotionData()

        # Extract values with defaults
        emotion.eye_blink_left = blendshapes.get(
            'eyeBlinkLeft', 0
        )
        emotion.eye_blink_right = blendshapes.get(
            'eyeBlinkRight', 0
        )
        emotion.jaw_open = blendshapes.get('jawOpen', 0)
        emotion.mouth_smile_left = blendshapes.get(
            'mouthSmileLeft', 0
        )
        emotion.mouth_smile_right = blendshapes.get(
            'mouthSmileRight', 0
        )
        emotion.brow_up_left = blendshapes.get(
            'browOuterUpLeft', 0
        )
        emotion.brow_up_right = blendshapes.get(
            'browOuterUpRight', 0
        )

        mouth_frown_left = blendshapes.get('mouthFrownLeft', 0)
        mouth_frown_right = blendshapes.get('mouthFrownRight', 0)

        # Computed scores
        emotion.mouth_openness = emotion.jaw_open
        emotion.smile_score = (
            emotion.mouth_smile_left +
            emotion.mouth_smile_right
        ) / 2
        emotion.brow_raise = (
            emotion.brow_up_left +
            emotion.brow_up_right
        ) / 2

        # Expression intensity (combined signal)
        intensity = (
            emotion.mouth_openness * 0.3 +
            emotion.smile_score * 0.3 +
            emotion.brow_raise * 0.2 +
            (mouth_frown_left + mouth_frown_right) / 2 * 0.2
        )

        # Track variance over time
        self.expression_history.append(intensity)
        if len(self.expression_history) > self.EXPRESSION_WINDOW:
            self.expression_history.pop(0)

        if len(self.expression_history) > 10:
            emotion.expression_variance = float(
                np.std(self.expression_history)
            )

        # Label
        if emotion.jaw_open > 0.4:
            emotion.expression_label = "surprised/vocalizing"
        elif emotion.smile_score > 0.4:
            emotion.expression_label = "smiling"
        elif (mouth_frown_left + mouth_frown_right) / 2 > 0.3:
            emotion.expression_label = "frowning"
        elif emotion.expression_variance < 0.02:
            emotion.expression_label = "flat_affect"
        else:
            emotion.expression_label = "neutral"

        return emotion

    def _draw_face_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        img_w: int,
        img_h: int
    ) -> np.ndarray:
        """
        Draw face landmarks manually on the frame
        (replaces mp.solutions.drawing_utils).
        """
        annotated = frame.copy()

        # Draw all landmarks as small dots
        for i, lm in enumerate(landmarks):
            x = int(lm[0] * img_w)
            y = int(lm[1] * img_h)
            
            # Color code by region
            if i in self.LEFT_EYE or i in self.RIGHT_EYE:
                color = (0, 255, 0)       # Green - eyes
                radius = 2
            elif i in self.LEFT_IRIS or i in self.RIGHT_IRIS:
                color = (255, 255, 0)     # Cyan - iris
                radius = 3
            elif i in [self.LEFT_MOUTH, self.RIGHT_MOUTH, 13, 14]:
                color = (0, 0, 255)       # Red - mouth
                radius = 2
            else:
                color = (200, 200, 200)   # Gray - other
                radius = 1

            cv2.circle(annotated, (x, y), radius, color, -1)

        # Draw eye contours
        for eye_indices in [self.LEFT_EYE, self.RIGHT_EYE]:
            pts = []
            for idx in eye_indices:
                x = int(landmarks[idx][0] * img_w)
                y = int(landmarks[idx][1] * img_h)
                pts.append([x, y])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(
                annotated, [pts], True, (0, 255, 0), 1
            )

        # Draw iris circles
        for iris_indices in [self.LEFT_IRIS, self.RIGHT_IRIS]:
            if max(iris_indices) < len(landmarks):
                iris_pts = landmarks[iris_indices]
                cx = int(iris_pts[:, 0].mean() * img_w)
                cy = int(iris_pts[:, 1].mean() * img_h)
                
                # Estimate radius
                radii = []
                for pt in iris_pts:
                    r = np.sqrt(
                        (pt[0] * img_w - cx) ** 2 +
                        (pt[1] * img_h - cy) ** 2
                    )
                    radii.append(r)
                radius = int(np.mean(radii))
                
                cv2.circle(
                    annotated, (cx, cy), radius,
                    (255, 255, 0), 1
                )
                cv2.circle(
                    annotated, (cx, cy), 2,
                    (0, 255, 255), -1
                )

        # Draw mouth
        mouth_indices = [
            61, 146, 91, 181, 84, 17,
            314, 405, 321, 375, 291
        ]
        mouth_pts = []
        for idx in mouth_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx][0] * img_w)
                y = int(landmarks[idx][1] * img_h)
                mouth_pts.append([x, y])
        if mouth_pts:
            pts = np.array(mouth_pts, dtype=np.int32)
            cv2.polylines(
                annotated, [pts], True, (0, 0, 255), 1
            )

        # Draw face oval (simplified jaw line)
        jaw_indices = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109
        ]
        jaw_pts = []
        for idx in jaw_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx][0] * img_w)
                y = int(landmarks[idx][1] * img_h)
                jaw_pts.append([x, y])
        if jaw_pts:
            pts = np.array(jaw_pts, dtype=np.int32)
            cv2.polylines(
                annotated, [pts], True, (150, 150, 150), 1
            )

        return annotated

    def analyze_frame(
        self,
        frame: np.ndarray
    ) -> FaceAnalysisResult:
        """
        Run complete face analysis on a single frame
        using the Tasks API.
        """
        result = FaceAnalysisResult(timestamp=time.time())
        img_h, img_w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)

        annotated = frame.copy()

        # Check if face was detected
        if (
            not detection_result.face_landmarks
            or len(detection_result.face_landmarks) == 0
        ):
            result.face_detected = False
            cv2.putText(
                annotated, "NO FACE DETECTED",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2
            )
            result.annotated_frame = annotated
            return result

        result.face_detected = True

        # Extract landmarks as numpy array
        face_landmarks = detection_result.face_landmarks[0]
        landmarks = np.array([
            (lm.x, lm.y, lm.z)
            for lm in face_landmarks
        ])
        result.landmarks_3d = landmarks

        # --- Blendshapes (Expression Analysis) ---
        blendshapes = self._extract_blendshapes(
            detection_result.face_blendshapes
        )

        # --- Head Pose from Transformation Matrix ---
        pitch, yaw, roll = self._extract_head_pose(
            detection_result.facial_transformation_matrixes
        )
        result.gaze.head_pose_pitch = pitch
        result.gaze.head_pose_yaw = yaw
        result.gaze.head_pose_roll = roll

        # --- EAR (Eye Aspect Ratio) ---
        left_ear = self._compute_ear(landmarks, self.LEFT_EYE)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EYE)
        result.gaze.left_ear = left_ear
        result.gaze.right_ear = right_ear

        # --- Blink Detection using Blendshapes ---
        blink_left = blendshapes.get('eyeBlinkLeft', 0)
        blink_right = blendshapes.get('eyeBlinkRight', 0)

        left_closed = blink_left > self.BLINK_THRESHOLD
        right_closed = blink_right > self.BLINK_THRESHOLD

        # Detect blink (transition from closed to open)
        if (
            self.left_eye_closed_prev
            and not left_closed
        ):
            self.blink_total += 1
        if (
            self.right_eye_closed_prev
            and not right_closed
        ):
            # Only count if left didn't already count
            # (avoid double counting)
            if not self.left_eye_closed_prev:
                self.blink_total += 1

        self.left_eye_closed_prev = left_closed
        self.right_eye_closed_prev = right_closed
        result.gaze.is_blinking = left_closed or right_closed

        # --- Gaze Direction ---
        direction, looking = self._compute_gaze_direction(
            landmarks, img_w, img_h, yaw, pitch
        )
        result.gaze.gaze_direction = direction
        result.gaze.is_looking_at_camera = looking

        # Track gaze-away events
        current_time = time.time()
        if not looking:
            if self.gaze_away_start is None:
                self.gaze_away_start = current_time
        else:
            if self.gaze_away_start is not None:
                duration = current_time - self.gaze_away_start
                self.total_gaze_away_time += duration
                if duration > 3.0:
                    self.gaze_events.append({
                        "type": "extended_gaze_away",
                        "start": self.gaze_away_start,
                        "duration": round(duration, 1),
                        "direction": direction,
                        "session_time": round(
                            self.gaze_away_start -
                            self.start_time, 1
                        )
                    })
                self.gaze_away_start = None

        # --- Expression Analysis (Blendshapes) ---
        result.emotion = self._analyze_expression_blendshapes(
            blendshapes
        )

        # --- Draw Landmarks Manually ---
        annotated = self._draw_face_landmarks(
            frame, landmarks, img_w, img_h
        )

        # --- Overlay Info Text ---
        color = (0, 255, 0) if looking else (0, 0, 255)
        gaze_text = f"Gaze: {direction}"
        if looking:
            gaze_text += " [EYE CONTACT]"

        cv2.putText(
            annotated, gaze_text,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )
        cv2.putText(
            annotated,
            f"EAR: {(left_ear + right_ear) / 2:.2f} | "
            f"Blinks: {self.blink_total}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1
        )
        cv2.putText(
            annotated,
            f"Expr: {result.emotion.expression_label} | "
            f"Smile: {result.emotion.smile_score:.2f}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 200, 0), 1
        )
        cv2.putText(
            annotated,
            f"Head: P{pitch:.0f} Y{yaw:.0f} R{roll:.0f}",
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1
        )

        # Blendshape debug info
        cv2.putText(
            annotated,
            f"BlinkL:{blink_left:.2f} BlinkR:{blink_right:.2f} "
            f"Jaw:{blendshapes.get('jawOpen', 0):.2f}",
            (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (150, 150, 255), 1
        )

        result.annotated_frame = annotated
        return result

    def get_session_stats(self) -> dict:
        """Get accumulated statistics for the session."""
        elapsed = time.time() - self.start_time
        blinks_per_min = (
            (self.blink_total / elapsed) * 60
            if elapsed > 0 else 0
        )

        return {
            "total_blinks": self.blink_total,
            "blinks_per_minute": round(blinks_per_min, 1),
            "total_gaze_away_seconds": round(
                self.total_gaze_away_time, 1
            ),
            "gaze_away_percentage": round(
                (self.total_gaze_away_time / elapsed) * 100, 1
            ) if elapsed > 0 else 0,
            "gaze_events": self.gaze_events,
            "expression_variance": round(
                float(np.std(self.expression_history)), 4
            ) if self.expression_history else 0,
            "session_duration_seconds": round(elapsed, 1)
        }

    def reset(self):
        """Reset all tracking for a new session."""
        self.blink_total = 0
        self.left_eye_closed_prev = False
        self.right_eye_closed_prev = False
        self.gaze_away_start = None
        self.total_gaze_away_time = 0.0
        self.gaze_events = []
        self.expression_history = []
        self.start_time = time.time()

    def close(self):
        """Clean up the detector."""
        if self.detector:
            self.detector.close()