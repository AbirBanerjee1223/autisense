# modules/dual_camera.py

import cv2
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CameraFrame:
    """Container for a camera frame with metadata."""
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    is_valid: bool = False
    source: str = ""


class CameraStream:
    """
    Threaded camera stream reader.
    Runs capture in a background thread so the main
    loop never blocks waiting for frames.
    """

    def __init__(self, source, name: str = "Camera"):
        self.source = source
        self.name = name
        self.capture: Optional[cv2.VideoCapture] = None
        self.latest_frame = CameraFrame(source=name)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Open the camera and start the reader thread."""
        try:
            self.capture = cv2.VideoCapture(self.source)

            if not self.capture.isOpened():
                print(f"[{self.name}] Failed to open: {self.source}")
                return False

            # Optimize capture settings
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if isinstance(self.source, int):
                # Laptop webcam optimizations
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv2.CAP_PROP_FPS, 30)

            self.is_running = True
            self._thread = threading.Thread(
                target=self._read_loop,
                daemon=True
            )
            self._thread.start()
            print(f"[{self.name}] Started successfully")
            return True

        except Exception as e:
            print(f"[{self.name}] Error starting: {e}")
            return False

    def _read_loop(self):
        """Continuously read frames in background."""
        while self.is_running:
            try:
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    with self._lock:
                        self.latest_frame = CameraFrame(
                            frame=frame.copy(),
                            timestamp=time.time(),
                            is_valid=True,
                            source=self.name
                        )
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def get_frame(self) -> CameraFrame:
        """Get the most recent frame (thread-safe)."""
        with self._lock:
            return self.latest_frame

    def stop(self):
        """Stop the camera stream."""
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
        print(f"[{self.name}] Stopped")


class DualCameraSystem:
    """
    Manages two simultaneous camera feeds:
    - Face Camera (laptop webcam): For gaze, expression, blink analysis
    - Body Camera (phone via IP Webcam): For posture, rocking, flapping
    """

    def __init__(
        self,
        laptop_source: int = 0,
        phone_url: Optional[str] = None
    ):
        self.face_cam = CameraStream(laptop_source, "FaceCam")
        self.body_cam = (
            CameraStream(phone_url, "BodyCam") if phone_url else None
        )
        self.is_dual_mode = False

    def start(self) -> dict:
        """Start all available cameras. Returns status dict."""
        status = {
            "face_cam": False,
            "body_cam": False,
            "mode": "single"
        }

        status["face_cam"] = self.face_cam.start()

        if self.body_cam is not None:
            status["body_cam"] = self.body_cam.start()
            if status["body_cam"]:
                self.is_dual_mode = True
                status["mode"] = "dual"

        return status

    def get_frames(self) -> Tuple[CameraFrame, Optional[CameraFrame]]:
        """Get latest frames from both cameras."""
        face_frame = self.face_cam.get_frame()
        body_frame = (
            self.body_cam.get_frame()
            if self.body_cam and self.is_dual_mode
            else None
        )
        return face_frame, body_frame

    def stop(self):
        """Stop all cameras."""
        self.face_cam.stop()
        if self.body_cam:
            self.body_cam.stop()


def create_dual_view(
    face_frame: np.ndarray,
    body_frame: Optional[np.ndarray],
    target_height: int = 480
) -> np.ndarray:
    """
    Create a side-by-side view of both camera feeds.
    Handles the case where body cam might not be available.
    """
    # Resize face frame
    h, w = face_frame.shape[:2]
    scale = target_height / h
    face_resized = cv2.resize(
        face_frame,
        (int(w * scale), target_height)
    )

    if body_frame is not None:
        h2, w2 = body_frame.shape[:2]
        scale2 = target_height / h2
        body_resized = cv2.resize(
            body_frame,
            (int(w2 * scale2), target_height)
        )

        # Add labels
        cv2.putText(
            face_resized, "FACE VIEW",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )
        cv2.putText(
            body_resized, "BODY VIEW",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 165, 255), 2
        )

        # Separator line
        separator = np.ones(
            (target_height, 3, 3), dtype=np.uint8
        ) * 128

        combined = np.hstack([
            face_resized, separator, body_resized
        ])
    else:
        cv2.putText(
            face_resized, "FACE VIEW (Single Mode)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )
        combined = face_resized

    return combined