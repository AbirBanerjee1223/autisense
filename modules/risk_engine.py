# modules/risk_engine.py

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from config import (
    EVIDENCE_DIR,
    GAZE_AWAY_DURATION_FLAG,
    BLINK_RATE_LOW,
    BLINK_RATE_HIGH,
    REPETITIVE_MOTION_THRESHOLD,
    EMOTION_FLAT_THRESHOLD
)


@dataclass
class EvidenceItem:
    """A single piece of evidence for the diagnostic report."""
    timestamp: float = 0.0
    session_time_str: str = ""
    category: str = ""            # gaze, motor, expression, blink
    description: str = ""
    confidence: float = 0.0
    severity: str = "low"         # low, medium, high
    screenshot_path: Optional[str] = None
    metric_name: str = ""
    metric_value: float = 0.0
    threshold_value: float = 0.0


@dataclass
class RiskAssessment:
    """Complete risk assessment for a session."""
    overall_risk_score: float = 0.0       # 0-100
    risk_level: str = "Low"               # Low, Moderate, High
    domain_scores: dict = field(default_factory=dict)
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""


class RiskEngine:
    """
    Explainable AI Risk Scoring System.
    Aggregates evidence from face and body analyzers,
    generates human-readable explanations for every flag.
    """

    def __init__(self, session_start_time: float):
        self.session_start = session_start_time
        self.evidence: List[EvidenceItem] = []
        self.frame_count = 0

        # Running tallies
        self.gaze_away_frames = 0
        self.total_face_frames = 0
        self.flat_affect_frames = 0
        self.rocking_frames = 0
        self.flapping_frames = 0
        self.total_body_frames = 0

        # Cooldowns (prevent duplicate evidence)
        self._last_gaze_flag = 0.0
        self._last_motor_flag = 0.0
        self._last_expression_flag = 0.0
        self.COOLDOWN_SECONDS = 5.0

    def _get_session_time(self) -> str:
        """Get formatted session timestamp."""
        elapsed = time.time() - self.session_start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins:02d}:{secs:02d}"

    def _save_evidence_screenshot(
        self,
        frame: np.ndarray,
        category: str,
        annotation: str = ""
    ) -> str:
        """Save a flagged frame as evidence."""
        timestamp = time.time()
        filename = (
            f"evidence_{category}_"
            f"{int(timestamp * 1000)}.jpg"
        )
        filepath = EVIDENCE_DIR / filename

        # Add annotation overlay
        evidence_frame = frame.copy()
        cv2.putText(
            evidence_frame,
            f"[{self._get_session_time()}] {annotation}",
            (10, evidence_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

        # Add red border to indicate flagged frame
        border_size = 3
        evidence_frame = cv2.copyMakeBorder(
            evidence_frame,
            border_size, border_size,
            border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 255)
        )

        cv2.imwrite(str(filepath), evidence_frame)
        return str(filepath)

    def process_face_result(
        self,
        face_result,  # FaceAnalysisResult
        raw_frame: np.ndarray
    ):
        """
        Process face analysis results and flag any
        atypical behaviors as evidence items.
        """
        if not face_result.face_detected:
            return

        self.total_face_frames += 1
        current_time = time.time()

        # --- Gaze Avoidance ---
        if not face_result.gaze.is_looking_at_camera:
            self.gaze_away_frames += 1

        gaze_away_pct = (
            (self.gaze_away_frames / max(self.total_face_frames, 1))
            * 100
        )

        # Flag extended gaze avoidance
        if (
            gaze_away_pct > 60
            and self.total_face_frames > 90
            and (current_time - self._last_gaze_flag)
            > self.COOLDOWN_SECONDS
        ):
            screenshot = self._save_evidence_screenshot(
                raw_frame, "gaze",
                f"Gaze Avoidance: {gaze_away_pct:.0f}%"
            )
            self.evidence.append(EvidenceItem(
                timestamp=current_time,
                session_time_str=self._get_session_time(),
                category="gaze",
                description=(
                    f"Subject avoided eye contact for "
                    f"{gaze_away_pct:.0f}% of session. "
                    f"Gaze direction: "
                    f"{face_result.gaze.gaze_direction}. "
                    f"Head yaw: "
                    f"{face_result.gaze.head_pose_yaw:.0f}°"
                ),
                confidence=min(gaze_away_pct / 100, 0.99),
                severity=(
                    "high" if gaze_away_pct > 80 else "medium"
                ),
                screenshot_path=screenshot,
                metric_name="gaze_away_percentage",
                metric_value=gaze_away_pct,
                threshold_value=60.0
            ))
            self._last_gaze_flag = current_time

        # --- Flat Affect Detection ---
        if face_result.emotion.expression_label == "flat_affect":
            self.flat_affect_frames += 1

        flat_pct = (
            (self.flat_affect_frames /
             max(self.total_face_frames, 1)) * 100
        )

        if (
            flat_pct > 70
            and self.total_face_frames > 90
            and (current_time - self._last_expression_flag)
            > self.COOLDOWN_SECONDS
        ):
            screenshot = self._save_evidence_screenshot(
                raw_frame, "expression",
                f"Flat Affect: variance="
                f"{face_result.emotion.expression_variance:.4f}"
            )
            self.evidence.append(EvidenceItem(
                timestamp=current_time,
                session_time_str=self._get_session_time(),
                category="expression",
                description=(
                    f"Subject displayed flat affect "
                    f"(minimal facial expression variation) "
                    f"for {flat_pct:.0f}% of session. "
                    f"Expression variance: "
                    f"{face_result.emotion.expression_variance:.4f} "
                    f"(typical >0.02)"
                ),
                confidence=min(flat_pct / 100, 0.99),
                severity="medium" if flat_pct < 85 else "high",
                screenshot_path=screenshot,
                metric_name="flat_affect_percentage",
                metric_value=flat_pct,
                threshold_value=70.0
            ))
            self._last_expression_flag = current_time

        # --- Atypical Blink Rate ---
        # (Checked in compute_assessment, needs session stats)

    def process_body_result(
        self,
        body_result,  # BodyMovementData
        raw_frame: np.ndarray
    ):
        """
        Process body analysis results and flag atypical
        motor behaviors as evidence items.
        """
        if not body_result.pose_detected:
            return

        self.total_body_frames += 1
        current_time = time.time()

        if body_result.is_rocking:
            self.rocking_frames += 1

        if body_result.is_hand_flapping:
            self.flapping_frames += 1

        # --- Flag Rocking ---
        rocking_pct = (
            (self.rocking_frames /
             max(self.total_body_frames, 1)) * 100
        )

        if (
            body_result.is_rocking
            and rocking_pct > 15
            and self.total_body_frames > 60
            and (current_time - self._last_motor_flag)
            > self.COOLDOWN_SECONDS
        ):
            screenshot = self._save_evidence_screenshot(
                raw_frame, "motor_rocking",
                f"Body Rocking @ {body_result.rocking_frequency:.1f}Hz"
            )
            self.evidence.append(EvidenceItem(
                timestamp=current_time,
                session_time_str=self._get_session_time(),
                category="motor",
                description=(
                    f"Repetitive body rocking detected at "
                    f"{body_result.rocking_frequency:.1f} Hz. "
                    f"Rocking present in {rocking_pct:.0f}% of "
                    f"frames. Repetitive motion autocorrelation "
                    f"score: "
                    f"{body_result.repetitive_motion_score:.2f}"
                ),
                confidence=min(
                    body_result.repetitive_motion_score + 0.3, 0.99
                ),
                severity=(
                    "high" if rocking_pct > 40 else "medium"
                ),
                screenshot_path=screenshot,
                metric_name="rocking_percentage",
                metric_value=rocking_pct,
                threshold_value=15.0
            ))
            self._last_motor_flag = current_time

        # --- Flag Hand Flapping ---
        flapping_pct = (
            (self.flapping_frames /
             max(self.total_body_frames, 1)) * 100
        )

        if (
            body_result.is_hand_flapping
            and flapping_pct > 10
            and self.total_body_frames > 60
            and (current_time - self._last_motor_flag)
            > self.COOLDOWN_SECONDS
        ):
            screenshot = self._save_evidence_screenshot(
                raw_frame, "motor_flapping",
                f"Hand Flapping Score: "
                f"{body_result.hand_flap_score:.2f}"
            )
            self.evidence.append(EvidenceItem(
                timestamp=current_time,
                session_time_str=self._get_session_time(),
                category="motor",
                description=(
                    f"Hand flapping behavior detected. "
                    f"Flap score: "
                    f"{body_result.hand_flap_score:.2f}. "
                    f"Present in {flapping_pct:.0f}% of frames. "
                    f"Characterized by rapid oscillatory wrist "
                    f"movements with high direction-reversal rate."
                ),
                confidence=min(
                    body_result.hand_flap_score + 0.2, 0.99
                ),
                severity=(
                    "high" if flapping_pct > 25 else "medium"
                ),
                screenshot_path=screenshot,
                metric_name="flapping_percentage",
                metric_value=flapping_pct,
                threshold_value=10.0
            ))
            self._last_motor_flag = current_time

    def compute_assessment(
        self,
        face_stats: dict,
        body_stats: Optional[dict] = None
    ) -> RiskAssessment:
        """
        Compute the final risk assessment combining all
        evidence from face and body analysis.

        Domain Scoring:
        - Social Attention (gaze, eye contact)    : 0-100
        - Facial Expression (affect range)        : 0-100
        - Motor Behavior (repetitive movements)   : 0-100
        - Physiological (blink rate)              : 0-100

        Overall = weighted average of domain scores.
        """
        assessment = RiskAssessment()

        # ===== DOMAIN 1: Social Attention (40% weight) =====
        gaze_away_pct = face_stats.get("gaze_away_percentage", 0)
        gaze_events_count = len(
            face_stats.get("gaze_events", [])
        )

        # Score: 0 = typical, 100 = highly atypical
        social_attention_score = 0.0
        if gaze_away_pct > 30:
            social_attention_score += min(
                (gaze_away_pct - 30) * 1.5, 70
            )
        social_attention_score += min(
            gaze_events_count * 8, 30
        )
        social_attention_score = min(social_attention_score, 100)

        # Add blink evidence if atypical
        bpm = face_stats.get("blinks_per_minute", 15)
        if bpm < BLINK_RATE_LOW or bpm > BLINK_RATE_HIGH:
            blink_evidence = EvidenceItem(
                timestamp=time.time(),
                session_time_str="SESSION",
                category="physiological",
                description=(
                    f"Atypical blink rate: {bpm:.1f} blinks/min. "
                    f"Typical range: {BLINK_RATE_LOW}-"
                    f"{BLINK_RATE_HIGH} blinks/min. "
                    f"{'Hypo-blinking may indicate ' if bpm < BLINK_RATE_LOW else 'Hyper-blinking may indicate '}"
                    f"{'sustained fixation or sensory processing differences.' if bpm < BLINK_RATE_LOW else 'anxiety or sensory sensitivity.'}"
                ),
                confidence=0.65,
                severity="low",
                metric_name="blinks_per_minute",
                metric_value=bpm,
                threshold_value=float(BLINK_RATE_LOW)
            )
            self.evidence.append(blink_evidence)

        # ===== DOMAIN 2: Facial Expression (25% weight) =====
        expression_var = face_stats.get(
            "expression_variance", 0.05
        )
        flat_pct = (
            (self.flat_affect_frames /
             max(self.total_face_frames, 1)) * 100
        )

        expression_score = 0.0
        if expression_var < 0.02:
            expression_score += min(
                (0.02 - expression_var) * 5000, 60
            )
        if flat_pct > 50:
            expression_score += min(
                (flat_pct - 50) * 0.8, 40
            )
        expression_score = min(expression_score, 100)

        # ===== DOMAIN 3: Motor Behavior (25% weight) =====
        motor_score = 0.0
        if body_stats:
            rocking_pct = body_stats.get(
                "rocking_percentage", 0
            )
            flapping_pct = body_stats.get(
                "flapping_percentage", 0
            )

            if rocking_pct > 5:
                motor_score += min(rocking_pct * 2, 50)
            if flapping_pct > 3:
                motor_score += min(flapping_pct * 2.5, 50)
            motor_score = min(motor_score, 100)

        # ===== DOMAIN 4: Physiological (10% weight) =====
        physio_score = 0.0
        if bpm < BLINK_RATE_LOW:
            physio_score = min(
                (BLINK_RATE_LOW - bpm) * 8, 100
            )
        elif bpm > BLINK_RATE_HIGH:
            physio_score = min(
                (bpm - BLINK_RATE_HIGH) * 5, 100
            )

        # ===== OVERALL RISK =====
        assessment.domain_scores = {
            "Social Attention": round(social_attention_score, 1),
            "Facial Expression": round(expression_score, 1),
            "Motor Behavior": round(motor_score, 1),
            "Physiological": round(physio_score, 1)
        }

        # Weighted average
        overall = (
            social_attention_score * 0.40 +
            expression_score * 0.25 +
            motor_score * 0.25 +
            physio_score * 0.10
        )
        assessment.overall_risk_score = round(
            min(overall, 100), 1
        )

        # Risk level
        if overall < 25:
            assessment.risk_level = "Low"
        elif overall < 50:
            assessment.risk_level = "Moderate"
        elif overall < 75:
            assessment.risk_level = "High"
        else:
            assessment.risk_level = "Very High"

        # Attach all collected evidence
        assessment.evidence_items = sorted(
            self.evidence,
            key=lambda e: e.timestamp
        )

        # ===== RECOMMENDATIONS =====
        recs = []
        if social_attention_score > 40:
            recs.append(
                "Refer for comprehensive eye contact and "
                "joint attention assessment by a developmental "
                "pediatrician."
            )
        if expression_score > 40:
            recs.append(
                "Evaluate for restricted affect range. "
                "Consider speech-language pathology assessment "
                "for social-emotional reciprocity."
            )
        if motor_score > 40:
            recs.append(
                "Refer for occupational therapy evaluation "
                "to assess repetitive motor behaviors and "
                "sensory processing."
            )
        if physio_score > 40:
            recs.append(
                "Note atypical blink patterns. Consider "
                "ophthalmological screening and sensory "
                "profile assessment."
            )
        if overall < 25:
            recs.append(
                "Current screening session shows typical "
                "behavioral patterns. Continue routine "
                "developmental monitoring."
            )

        assessment.recommendations = recs

        # ===== SUMMARY =====
        high_evidence = [
            e for e in self.evidence if e.severity == "high"
        ]
        med_evidence = [
            e for e in self.evidence if e.severity == "medium"
        ]

        assessment.summary = (
            f"Screening session analyzed "
            f"{face_stats.get('session_duration_seconds', 0):.0f} "
            f"seconds of behavioral data. "
            f"Overall risk score: "
            f"{assessment.overall_risk_score}/100 "
            f"({assessment.risk_level}). "
            f"{len(high_evidence)} high-severity and "
            f"{len(med_evidence)} medium-severity behavioral "
            f"markers were identified across "
            f"{len(set(e.category for e in self.evidence))} "
            f"domains. "
            f"This is a screening tool only and does NOT "
            f"constitute a clinical diagnosis."
        )

        return assessment

    def reset(self):
        """Reset all evidence for a new session."""
        self.evidence = []
        self.frame_count = 0
        self.gaze_away_frames = 0
        self.total_face_frames = 0
        self.flat_affect_frames = 0
        self.rocking_frames = 0
        self.flapping_frames = 0
        self.total_body_frames = 0
        self.session_start = time.time()
        self._last_gaze_flag = 0.0
        self._last_motor_flag = 0.0
        self._last_expression_flag = 0.0

        # Clear evidence directory
        for f in EVIDENCE_DIR.glob("evidence_*.jpg"):
            try:
                f.unlink()
            except Exception:
                pass