# modules/stimulus_engine.py

import time
import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from pathlib import Path
from modules.reciprocity_tracker import ReciprocityTracker
from config import (
    SOCIAL_GEOMETRIC_VIDEO,
    SMILE_PROMPT_VIDEO,
    NAME_CALL_AUDIO,
    BASELINES
)


class StimulusPhase(Enum):
    """The structured phases of the screening protocol."""
    BASELINE = "baseline"              # 15s: Free viewing, no stimulus
    SOCIAL_GEOMETRIC = "social_geo"    # 30s: Split-screen preference test
    NAME_CALL = "name_call"            # 10s: Audio response latency test
    SMILE_PROMPT = "smile_prompt"      # 10s: Emotional reciprocity test
    COOLDOWN = "cooldown"              # 10s: Post-stimulus baseline
    COMPLETE = "complete"


@dataclass
class StimulusEvent:
    """A timestamped event during stimulus presentation."""
    timestamp: float = 0.0
    session_time: float = 0.0
    phase: str = ""
    event_type: str = ""       # stimulus_start, audio_played, phase_change
    description: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class SocialGeoMetrics:
    """Metrics from the social vs geometric preference test."""
    total_frames: int = 0
    gaze_left_frames: int = 0      # Looking at social side
    gaze_right_frames: int = 0     # Looking at geometric side
    gaze_center_frames: int = 0    # Looking at center/camera
    gaze_away_frames: int = 0      # Not looking at screen at all
    social_preference_pct: float = 0.0
    geometric_preference_pct: float = 0.0


@dataclass
class NameCallMetrics:
    """Metrics from the name-call response test."""
    audio_played_time: float = 0.0
    head_movement_detected_time: float = 0.0
    response_latency_ms: float = -1.0  # -1 = no response
    pre_call_head_yaw: float = 0.0
    post_call_head_yaw: float = 0.0
    responded: bool = False


@dataclass 
class ReciprocityMetrics:
    """Metrics from the smile reciprocity test."""
    smile_prompt_time: float = 0.0
    total_frames: int = 0
    smile_detected_frames: int = 0
    smile_reciprocity_pct: float = 0.0
    peak_smile_score: float = 0.0
    latency_to_first_smile_ms: float = -1.0
    first_smile_detected: bool = False


class StimulusEngine:
    """
    Controls the clinical screening protocol.

    Protocol Structure (90 seconds total):
    ─────────────────────────────────────
    Phase 1: BASELINE       (0-15s)    Free viewing
    Phase 2: SOCIAL_GEO     (15-45s)   Split screen preference
    Phase 3: NAME_CALL      (45-55s)   Audio response latency
    Phase 4: SMILE_PROMPT   (55-75s)   Emotional reciprocity
    Phase 5: COOLDOWN       (75-90s)   Post-stimulus baseline
    """

    # Phase timing (seconds from session start)
    PHASE_SCHEDULE = [
        (0,  StimulusPhase.BASELINE),
        (15, StimulusPhase.SOCIAL_GEOMETRIC),
        (45, StimulusPhase.NAME_CALL),
        (55, StimulusPhase.SMILE_PROMPT),
        (75, StimulusPhase.COOLDOWN),
        (90, StimulusPhase.COMPLETE),
    ]

    def __init__(self):
        self.session_start: float = 0.0
        self.current_phase = StimulusPhase.BASELINE
        self.events: List[StimulusEvent] = []
        self.reciprocity_tracker = ReciprocityTracker()

        # Video readers
        self._social_geo_cap: Optional[cv2.VideoCapture] = None
        self._smile_cap: Optional[cv2.VideoCapture] = None

        # Phase-specific metrics
        self.social_geo_metrics = SocialGeoMetrics()
        self.name_call_metrics = NameCallMetrics()
        self.reciprocity_metrics = ReciprocityMetrics()

        # State tracking
        self._name_call_audio_played = False
        self._name_call_monitoring = False
        self._pre_call_yaw_samples: List[float] = []
        self._smile_prompt_started = False

        # Current stimulus frame to display
        self.current_stimulus_frame: Optional[np.ndarray] = None
        self.should_play_audio = False  # Flag for UI to play audio

    def start_session(self):
        """Begin the screening protocol."""
        self.session_start = time.time()
        self.current_phase = StimulusPhase.BASELINE
        self.reciprocity_tracker.start_session()

        # Open video files
        if Path(SOCIAL_GEOMETRIC_VIDEO).exists():
            self._social_geo_cap = cv2.VideoCapture(
                SOCIAL_GEOMETRIC_VIDEO
            )
        if Path(SMILE_PROMPT_VIDEO).exists():
            self._smile_cap = cv2.VideoCapture(
                SMILE_PROMPT_VIDEO
            )

        self._log_event(
            "session_start",
            "Screening protocol initiated"
        )

    def get_elapsed(self) -> float:
        """Seconds since session start."""
        return time.time() - self.session_start

    def get_current_phase(self) -> StimulusPhase:
        """Determine which phase we're in based on elapsed time."""
        elapsed = self.get_elapsed()

        new_phase = StimulusPhase.BASELINE
        for phase_time, phase in self.PHASE_SCHEDULE:
            if elapsed >= phase_time:
                new_phase = phase

        # Log phase transitions
        if new_phase != self.current_phase:
            self._log_event(
                "phase_change",
                f"Transitioned to {new_phase.value}",
                {"from": self.current_phase.value,
                 "to": new_phase.value}
            )
            self.current_phase = new_phase

        return self.current_phase

    def update(
        self,
        gaze_direction: str,
        is_looking: bool,
        head_yaw: float,
        smile_score: float,
    ) -> dict:
        """
        Called every frame. Updates metrics based on current
        phase and behavioral data.

        Returns dict with:
        - phase: current phase name
        - instruction: text to show the user/clinician
        - stimulus_frame: video frame to display (or None)
        - play_audio: whether to trigger audio this frame
        """
        phase = self.get_current_phase()
        elapsed = self.get_elapsed()

        result = {
            "phase": phase.value,
            "instruction": "",
            "stimulus_frame": None,
            "play_audio": False,
            "phase_progress": 0.0,
        }

        # ===== PHASE 1: BASELINE =====
        if phase == StimulusPhase.BASELINE:
            result["instruction"] = (
                "BASELINE: Observing natural behavior. "
                "No stimulus presented."
            )
            progress = min(elapsed / 15.0, 1.0)
            result["phase_progress"] = progress

            # Show a neutral gray screen as "stimulus"
            self.current_stimulus_frame = self._create_instruction_frame(
                "Baseline Phase",
                f"Observing natural behavior... ({15 - elapsed:.0f}s)",
                (200, 200, 200)
            )
            result["stimulus_frame"] = self.current_stimulus_frame

        # ===== PHASE 2: SOCIAL VS GEOMETRIC =====
        elif phase == StimulusPhase.SOCIAL_GEOMETRIC:
            result["instruction"] = (
                "SOCIAL/GEOMETRIC TEST: Tracking visual preference. "
                "Subject views split-screen stimulus."
            )
            phase_elapsed = elapsed - 15.0
            progress = min(phase_elapsed / 30.0, 1.0)
            result["phase_progress"] = progress

            # Read video frame
            if self._social_geo_cap and self._social_geo_cap.isOpened():
                ret, frame = self._social_geo_cap.read()
                if ret:
                    self.current_stimulus_frame = frame
                else:
                    # Loop video
                    self._social_geo_cap.set(
                        cv2.CAP_PROP_POS_FRAMES, 0
                    )
                    ret, frame = self._social_geo_cap.read()
                    if ret:
                        self.current_stimulus_frame = frame
            
            result["stimulus_frame"] = self.current_stimulus_frame

            # Track gaze preference
            self.social_geo_metrics.total_frames += 1

            if not is_looking:
                self.social_geo_metrics.gaze_away_frames += 1
            elif gaze_direction == "left":
                # Left side = social stimulus
                self.social_geo_metrics.gaze_left_frames += 1
            elif gaze_direction == "right":
                # Right side = geometric stimulus
                self.social_geo_metrics.gaze_right_frames += 1
            else:
                self.social_geo_metrics.gaze_center_frames += 1

            # Update preference percentages
            looking_frames = max(
                self.social_geo_metrics.gaze_left_frames +
                self.social_geo_metrics.gaze_right_frames, 1
            )
            self.social_geo_metrics.social_preference_pct = (
                self.social_geo_metrics.gaze_left_frames /
                looking_frames * 100
            )
            self.social_geo_metrics.geometric_preference_pct = (
                self.social_geo_metrics.gaze_right_frames /
                looking_frames * 100
            )

        # ===== PHASE 3: NAME CALL =====
        elif phase == StimulusPhase.NAME_CALL:
            phase_elapsed = elapsed - 45.0

            if not self._name_call_audio_played:
                # Collect pre-call baseline yaw
                if phase_elapsed < 1.0:
                    self._pre_call_yaw_samples.append(head_yaw)
                    result["instruction"] = (
                        "NAME-CALL TEST: Preparing audio stimulus..."
                    )
                elif phase_elapsed >= 1.0:
                    # Fire the audio
                    result["play_audio"] = True
                    self.should_play_audio = True
                    self._name_call_audio_played = True
                    self._name_call_monitoring = True

                    self.name_call_metrics.audio_played_time = time.time()
                    if self._pre_call_yaw_samples:
                        self.name_call_metrics.pre_call_head_yaw = (
                            np.mean(self._pre_call_yaw_samples)
                        )

                    self._log_event(
                        "audio_played",
                        "Name-call audio stimulus triggered",
                        {"pre_yaw": self.name_call_metrics.pre_call_head_yaw}
                    )
                    result["instruction"] = (
                        "NAME-CALL TEST: Audio played! "
                        "Measuring response latency..."
                    )
            else:
                result["instruction"] = (
                    "NAME-CALL TEST: Monitoring head turn response..."
                )

            # Monitor for head turn response
            if (
                self._name_call_monitoring
                and not self.name_call_metrics.responded
            ):
                yaw_change = abs(
                    head_yaw -
                    self.name_call_metrics.pre_call_head_yaw
                )

                # Significant head turn detected (>12 degrees)
                if yaw_change > 12.0:
                    response_time = time.time()
                    latency_ms = (
                        (response_time -
                         self.name_call_metrics.audio_played_time)
                        * 1000
                    )

                    self.name_call_metrics.responded = True
                    self.name_call_metrics.head_movement_detected_time = (
                        response_time
                    )
                    self.name_call_metrics.response_latency_ms = latency_ms
                    self.name_call_metrics.post_call_head_yaw = head_yaw

                    self._log_event(
                        "name_call_response",
                        f"Head turn detected. Latency: {latency_ms:.0f}ms",
                        {"latency_ms": latency_ms,
                         "yaw_change": yaw_change}
                    )

            progress = min(phase_elapsed / 10.0, 1.0)
            result["phase_progress"] = progress

            # Show instruction screen during name call
            status = (
                f"Latency: {self.name_call_metrics.response_latency_ms:.0f}ms"
                if self.name_call_metrics.responded
                else "Waiting for response..."
            )
            self.current_stimulus_frame = self._create_instruction_frame(
                "Name-Call Response Test",
                status,
                (180, 220, 180) if self.name_call_metrics.responded
                else (220, 200, 180)
            )
            result["stimulus_frame"] = self.current_stimulus_frame

        # ===== PHASE 4: SMILE PROMPT =====
        elif phase == StimulusPhase.SMILE_PROMPT:
            result["instruction"] = (
                "RECIPROCITY TEST: Presenting smile stimulus. "
                "Monitoring for emotional mirroring."
            )
            phase_elapsed = elapsed - 55.0
            progress = min(phase_elapsed / 20.0, 1.0)
            result["phase_progress"] = progress

            if not self._smile_prompt_started:
                self._smile_prompt_started = True
                self.reciprocity_metrics.smile_prompt_time = time.time()

            # Read smile video frame
            if self._smile_cap and self._smile_cap.isOpened():
                ret, frame = self._smile_cap.read()
                if ret:
                    self.current_stimulus_frame = frame
                else:
                    self._smile_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._smile_cap.read()
                    if ret:
                        self.current_stimulus_frame = frame

            result["stimulus_frame"] = self.current_stimulus_frame

            # Calculate stimulus smile intensity
            # (smile starts at 3s into the video, peaks at 5s)
            stimulus_smile_intensity = 0.0
            if phase_elapsed > 3.0:
                stimulus_smile_intensity = min(
                    (phase_elapsed - 3.0) / 2.0, 1.0
                )

            # Activate reciprocity tracker
            self.reciprocity_tracker.set_prompt_active(
                active=True,
                stimulus_smile_intensity=stimulus_smile_intensity
            )

            # Process through reciprocity tracker
            # (smile_score comes from the function parameter)
            recip_result = self.reciprocity_tracker.process_frame(
                smile_score=smile_score,
                brow_raise_score=0.0,  # Can be passed if available
                jaw_open_score=0.0,
                expression_label="smiling" if smile_score > 0.3 else "neutral",
                stimulus_intensity=stimulus_smile_intensity,
            )

            # Also update legacy metrics
            self.reciprocity_metrics.total_frames += 1

            if smile_score > 0.3:
                self.reciprocity_metrics.smile_detected_frames += 1
                self.reciprocity_metrics.peak_smile_score = max(
                    self.reciprocity_metrics.peak_smile_score,
                    smile_score
                )

                if not self.reciprocity_metrics.first_smile_detected:
                    self.reciprocity_metrics.first_smile_detected = True
                    latency = (
                        (time.time() -
                         self.reciprocity_metrics.smile_prompt_time)
                        * 1000
                    )
                    self.reciprocity_metrics.latency_to_first_smile_ms = (
                        latency
                    )
                    self._log_event(
                        "smile_reciprocity",
                        f"First smile detected. Latency: {latency:.0f}ms. "
                        f"Mirroring: {recip_result.is_mirroring}",
                        {"smile_score": smile_score,
                         "latency_ms": latency,
                         "is_mirroring": recip_result.is_mirroring}
                    )

            if self.reciprocity_metrics.total_frames > 0:
                self.reciprocity_metrics.smile_reciprocity_pct = (
                    self.reciprocity_metrics.smile_detected_frames /
                    self.reciprocity_metrics.total_frames * 100
                )

        # ===== PHASE 5: COOLDOWN =====
        elif phase == StimulusPhase.COOLDOWN:
            self.reciprocity_tracker.set_prompt_active(False)
            result["instruction"] = (
                "COOLDOWN: Post-stimulus observation period."
            )
            phase_elapsed = elapsed - 75.0
            progress = min(phase_elapsed / 15.0, 1.0)
            result["phase_progress"] = progress

            self.current_stimulus_frame = self._create_instruction_frame(
                "Cooldown",
                f"Session ending in {max(0, 90 - elapsed):.0f}s",
                (200, 210, 220)
            )
            result["stimulus_frame"] = self.current_stimulus_frame

        # ===== COMPLETE =====
        elif phase == StimulusPhase.COMPLETE:
            result["instruction"] = "SESSION COMPLETE"
            result["phase_progress"] = 1.0

        return result

    def _create_instruction_frame(
        self,
        title: str,
        subtitle: str,
        bg_color: tuple = (200, 200, 200)
    ) -> np.ndarray:
        """Create a simple instruction display frame."""
        frame = np.ones((480, 640, 3), dtype=np.uint8)
        frame[:] = bg_color

        cv2.putText(
            frame, title,
            (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (50, 50, 50), 2
        )
        cv2.putText(
            frame, subtitle,
            (50, 260), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (80, 80, 80), 1
        )
        return frame

    def _log_event(
        self, event_type: str, description: str,
        data: dict = None
    ):
        """Log a stimulus event."""
        self.events.append(StimulusEvent(
            timestamp=time.time(),
            session_time=self.get_elapsed(),
            phase=self.current_phase.value,
            event_type=event_type,
            description=description,
            data=data or {}
        ))

    def get_all_metrics(self) -> dict:
        """Get all stimulus-response metrics for reporting."""
        
        # Get detailed reciprocity report
        recip_report = self.reciprocity_tracker.compute_report()
        
        return {
            "social_geometric": {
                "total_frames": self.social_geo_metrics.total_frames,
                "social_preference_pct": round(
                    self.social_geo_metrics.social_preference_pct, 1
                ),
                "geometric_preference_pct": round(
                    self.social_geo_metrics.geometric_preference_pct, 1
                ),
                "gaze_away_pct": round(
                    (self.social_geo_metrics.gaze_away_frames /
                     max(self.social_geo_metrics.total_frames, 1))
                    * 100, 1
                ),
            },
            "name_call": {
                "responded": self.name_call_metrics.responded,
                "latency_ms": round(
                    self.name_call_metrics.response_latency_ms, 0
                ),
                "pre_yaw": round(
                    self.name_call_metrics.pre_call_head_yaw, 1
                ),
                "post_yaw": round(
                    self.name_call_metrics.post_call_head_yaw, 1
                ),
            },
            "reciprocity": {
                "smile_reciprocity_pct": round(
                    recip_report.smile_reciprocity_pct, 1
                ),
                "peak_smile_score": round(
                    recip_report.peak_smile_intensity, 3
                ),
                "first_smile_latency_ms": round(
                    recip_report.mean_smile_latency_ms, 0
                ),
                "responded": self.reciprocity_metrics.first_smile_detected,
                # NEW: Advanced metrics from reciprocity tracker
                "contextual_smile_pct": round(
                    recip_report.contextual_smile_pct, 1
                ),
                "emotional_congruence": round(
                    recip_report.emotional_congruence_score, 3
                ),
                "affect_synchrony": round(
                    recip_report.affect_synchrony_score, 3
                ),
                "expression_diversity": round(
                    recip_report.expression_diversity, 3
                ),
                "mirroring_frequency": round(
                    recip_report.mirroring_frequency, 1
                ),
                "smile_episodes": len(recip_report.smile_events),
                "mean_smile_duration_ms": round(
                    recip_report.mean_smile_duration_ms, 0
                ),
            },
            "events": [
                {
                    "time": round(e.session_time, 1),
                    "phase": e.phase,
                    "type": e.event_type,
                    "desc": e.description
                }
                for e in self.events
            ]
        }

    def cleanup(self):
        """Release video captures."""
        if self._social_geo_cap:
            self._social_geo_cap.release()
        if self._smile_cap:
            self._smile_cap.release()