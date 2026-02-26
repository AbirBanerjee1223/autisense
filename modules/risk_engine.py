# modules/risk_engine.py (COMPLETE FILE - ALL FIXES APPLIED)

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from config import EVIDENCE_DIR, BASELINES


# =============================================
# DATA CLASSES
# =============================================

@dataclass
class EvidenceItem:
    """A single piece of clinical evidence."""
    timestamp: float = 0.0
    session_time_str: str = ""
    category: str = ""
    description: str = ""
    confidence: float = 0.0
    severity: str = "low"
    screenshot_path: Optional[str] = None
    metric_name: str = ""
    metric_value: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    z_score: float = 0.0


@dataclass
class ClinicalDeviation:
    """A single domain's deviation from neurotypical baseline."""
    domain_name: str = ""
    dsm5_code: str = ""
    metric_value: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    z_score: float = 0.0
    interpretation: str = ""
    clinical_significance: str = ""  # typical, borderline, atypical


@dataclass
class RiskAssessment:
    """Complete clinical assessment for a session."""
    deviations: List[ClinicalDeviation] = field(default_factory=list)
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""
    overall_risk_score: float = 0.0
    risk_level: str = "Typical"
    domain_scores: dict = field(default_factory=dict)


# =============================================
# HELPER FUNCTIONS (FIXED)
# =============================================

def compute_z_score(value: float, mean: float, std: float) -> float:
    """
    Compute standard deviation distance from baseline.
    Clamped to [-4, +4] to prevent absurd outliers
    from data collection artifacts.
    """
    if std == 0:
        return 0.0
    z = (value - mean) / std
    return max(-4.0, min(4.0, z))


def interpret_z_score(
    z: float,
    higher_is_worse: bool = True
) -> str:
    """
    Interpret a z-score clinically WITH PROPER DIRECTIONALITY.

    Args:
        z: The z-score value
        higher_is_worse: 
            True  = Only POSITIVE z is concerning
                    (e.g., more gaze away, longer latency)
            False = Only NEGATIVE z is concerning
                    (e.g., less social preference, less smiling)

    Examples:
        Gaze away 3% when baseline is 25%:
            z = -2.2, higher_is_worse=True -> "typical"
            (Less gaze away = GOOD, not flagged)

        Gaze away 50% when baseline is 25%:
            z = +2.5, higher_is_worse=True -> "atypical"
            (More gaze away = BAD, flagged)

        Social preference 30% when baseline is 70%:
            z = -3.3, higher_is_worse=False -> "atypical"
            (Less social preference = BAD, flagged)
    """
    if higher_is_worse:
        # Only flag when value is ABOVE baseline (positive z)
        if z > 2.0:
            return "atypical"
        elif z > 1.0:
            return "borderline"
        else:
            return "typical"
    else:
        # Only flag when value is BELOW baseline (negative z)
        if z < -2.0:
            return "atypical"
        elif z < -1.0:
            return "borderline"
        else:
            return "typical"


def interpret_z_score_bilateral(z: float) -> str:
    """
    For metrics where BOTH high and low are concerning.
    (e.g., blink rate: too low OR too high = atypical)
    """
    z_abs = abs(z)
    if z_abs > 2.0:
        return "atypical"
    elif z_abs > 1.0:
        return "borderline"
    else:
        return "typical"


# =============================================
# RISK ENGINE
# =============================================

class RiskEngine:
    """
    Clinical Risk Engine using Standard Deviation analysis.

    Instead of arbitrary 0-100 scores, computes how many
    standard deviations each behavioral metric deviates
    from published neurotypical baselines.

    Directionality is handled properly:
    - Gaze away: only HIGH values are concerning
    - Social preference: only LOW values are concerning
    - Blink rate: BOTH extremes are concerning
    """

    def __init__(self, session_start_time: float):
        self.session_start = session_start_time
        self.evidence: List[EvidenceItem] = []

        # Running tallies
        self.gaze_away_frames = 0
        self.total_face_frames = 0
        self.flat_affect_frames = 0

        # Cooldowns to prevent duplicate evidence
        self._last_gaze_flag = 0.0
        self._last_expression_flag = 0.0
        self._last_motor_flag = 0.0
        self.COOLDOWN = 5.0

    def _session_time_str(self) -> str:
        """Get formatted session timestamp MM:SS."""
        elapsed = time.time() - self.session_start
        return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

    def _save_screenshot(
        self, frame: np.ndarray, category: str, text: str
    ) -> str:
        """Save a flagged frame as evidence with annotation."""
        ts = int(time.time() * 1000)
        filename = f"evidence_{category}_{ts}.jpg"
        filepath = EVIDENCE_DIR / filename

        ev_frame = frame.copy()
        cv2.putText(
            ev_frame,
            f"[{self._session_time_str()}] {text}",
            (10, ev_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        border = 3
        ev_frame = cv2.copyMakeBorder(
            ev_frame, border, border, border, border,
            cv2.BORDER_CONSTANT, value=(0, 0, 255)
        )
        cv2.imwrite(str(filepath), ev_frame)
        return str(filepath)

    # =============================================
    # FRAME-BY-FRAME PROCESSING
    # =============================================

    def process_face_result(self, face_result, raw_frame):
        """Process face analysis results and flag evidence."""
        if not face_result.face_detected:
            return

        self.total_face_frames += 1
        now = time.time()

        # Count gaze away frames
        if not face_result.gaze.is_looking_at_camera:
            self.gaze_away_frames += 1

        # Count flat affect frames
        if face_result.emotion.expression_label == "flat_affect":
            self.flat_affect_frames += 1

        # --- Flag gaze avoidance ---
        gaze_pct = (
            self.gaze_away_frames /
            max(self.total_face_frames, 1) * 100
        )

        # Only flag if gaze away is HIGH (above baseline)
        # and we have enough data (5+ seconds)
        if (
            gaze_pct > 50
            and self.total_face_frames > 150
            and (now - self._last_gaze_flag) > self.COOLDOWN
        ):
            z = compute_z_score(
                gaze_pct,
                BASELINES["gaze_away_pct"]["mean"],
                BASELINES["gaze_away_pct"]["std"]
            )

            # Only flag if z is POSITIVE (more gaze away than baseline)
            if z > 1.0:
                screenshot = self._save_screenshot(
                    raw_frame, "gaze",
                    f"Gaze away: {gaze_pct:.0f}% (z={z:+.1f})"
                )
                self.evidence.append(EvidenceItem(
                    timestamp=now,
                    session_time_str=self._session_time_str(),
                    category="gaze",
                    description=(
                        f"Gaze avoidance at {gaze_pct:.0f}% "
                        f"(baseline: "
                        f"{BASELINES['gaze_away_pct']['mean']}"
                        f"+-{BASELINES['gaze_away_pct']['std']}%). "
                        f"Z-score: {z:+.1f} SD above neurotypical mean. "
                        f"Elevated gaze avoidance per DSM-5-TR A.2."
                    ),
                    confidence=min(z / 3.0, 0.99),
                    severity="high" if z > 2 else "medium",
                    screenshot_path=screenshot,
                    metric_name="gaze_away_pct",
                    metric_value=gaze_pct,
                    baseline_mean=BASELINES["gaze_away_pct"]["mean"],
                    baseline_std=BASELINES["gaze_away_pct"]["std"],
                    z_score=z
                ))
                self._last_gaze_flag = now

        # --- Flag flat affect ---
        flat_pct = (
            self.flat_affect_frames /
            max(self.total_face_frames, 1) * 100
        )

        # Require 5+ seconds of data before flagging flat affect
        if (
            flat_pct > 60
            and self.total_face_frames > 150
            and (now - self._last_expression_flag) > self.COOLDOWN
        ):
            screenshot = self._save_screenshot(
                raw_frame, "expression",
                f"Flat affect: {flat_pct:.0f}%"
            )
            self.evidence.append(EvidenceItem(
                timestamp=now,
                session_time_str=self._session_time_str(),
                category="expression",
                description=(
                    f"Flat affect detected in {flat_pct:.0f}% "
                    f"of frames. Expression variance: "
                    f"{face_result.emotion.expression_variance:.4f}. "
                    f"Reduced affect range per DSM-5-TR A.2."
                ),
                confidence=min(flat_pct / 100, 0.99),
                severity="medium" if flat_pct < 80 else "high",
                screenshot_path=screenshot,
                metric_name="flat_affect_pct",
                metric_value=flat_pct,
            ))
            self._last_expression_flag = now

    def process_body_result(self, body_result, raw_frame):
        """Process body analysis results and flag motor evidence."""
        if not body_result.pose_detected:
            return

        now = time.time()

        # Flag rocking
        if (
            body_result.is_rocking
            and (now - self._last_motor_flag) > self.COOLDOWN
        ):
            screenshot = self._save_screenshot(
                raw_frame, "motor",
                f"Rocking @ {body_result.rocking_frequency:.1f}Hz"
            )
            self.evidence.append(EvidenceItem(
                timestamp=now,
                session_time_str=self._session_time_str(),
                category="motor",
                description=(
                    f"Repetitive body rocking at "
                    f"{body_result.rocking_frequency:.1f} Hz. "
                    f"Autocorrelation: "
                    f"{body_result.repetitive_motion_score:.2f}. "
                    f"DSM-5-TR criterion B.1 "
                    f"(Stereotyped motor movements)."
                ),
                confidence=min(
                    body_result.repetitive_motion_score + 0.3, 0.99
                ),
                severity="high",
                screenshot_path=screenshot,
                metric_name="rocking_frequency_hz",
                metric_value=body_result.rocking_frequency,
            ))
            self._last_motor_flag = now

        # Flag hand flapping
        if (
            body_result.is_hand_flapping
            and (now - self._last_motor_flag) > self.COOLDOWN
        ):
            screenshot = self._save_screenshot(
                raw_frame, "motor_flap",
                f"Flapping: {body_result.hand_flap_score:.2f}"
            )
            self.evidence.append(EvidenceItem(
                timestamp=now,
                session_time_str=self._session_time_str(),
                category="motor",
                description=(
                    f"Hand flapping detected. "
                    f"Flap score: {body_result.hand_flap_score:.2f}. "
                    f"Rapid oscillatory wrist movements. "
                    f"DSM-5-TR B.1."
                ),
                confidence=min(
                    body_result.hand_flap_score + 0.2, 0.99
                ),
                severity="high",
                screenshot_path=screenshot,
                metric_name="flap_score",
                metric_value=body_result.hand_flap_score,
            ))
            self._last_motor_flag = now

    # =============================================
    # FINAL ASSESSMENT (ALL FIXES APPLIED)
    # =============================================

    def compute_assessment(
        self,
        face_stats: dict,
        stimulus_metrics: Optional[dict] = None,
        body_stats: Optional[dict] = None,
    ) -> RiskAssessment:
        """
        Compute clinical assessment using standard deviations
        from published neurotypical baselines.

        DIRECTIONALITY IS HANDLED CORRECTLY:
        - Social Preference: LOW = concerning (higher_is_worse=False)
        - Name-Call Latency: HIGH = concerning (higher_is_worse=True)
        - Reciprocity: LOW = concerning (higher_is_worse=False)
        - Gaze Avoidance: HIGH = concerning (higher_is_worse=True)
        - Expression Variance: LOW = concerning (higher_is_worse=False)
        - Blink Rate: BOTH extremes = concerning (bilateral)
        """
        assessment = RiskAssessment()
        deviations = []

        # =================================================
        # DOMAIN 1: SOCIAL VISUAL PREFERENCE (DSM-5 A.1)
        # Lower social preference = more concerning
        # =================================================
        if stimulus_metrics and stimulus_metrics.get("social_geometric"):
            sg = stimulus_metrics["social_geometric"]
            social_pct = sg.get("social_preference_pct", 50)
            bl = BASELINES["social_preference_pct"]
            z = compute_z_score(social_pct, bl["mean"], bl["std"])

            deviations.append(ClinicalDeviation(
                domain_name="Social Visual Preference",
                dsm5_code="A.1 (Social-emotional reciprocity)",
                metric_value=social_pct,
                baseline_mean=bl["mean"],
                baseline_std=bl["std"],
                z_score=z,
                interpretation=(
                    f"Subject preferred social stimuli "
                    f"{social_pct:.0f}% of the time "
                    f"(baseline: {bl['mean']}+-{bl['std']}%). "
                    f"Deviation: {z:+.1f} SD. "
                    f"{'Below average - reduced social interest.' if z < -1 else 'Within typical range.'}"
                ),
                # LOW social preference = concerning
                clinical_significance=interpret_z_score(
                    z, higher_is_worse=False
                )
            ))

        # =================================================
        # DOMAIN 2: NAME-CALL RESPONSE LATENCY (DSM-5 A.1)
        # Higher latency = more concerning
        # No response = highly atypical
        # =================================================
        if stimulus_metrics and stimulus_metrics.get("name_call"):
            nc = stimulus_metrics["name_call"]

            if nc.get("responded"):
                latency = nc["latency_ms"]
                bl = BASELINES["name_call_latency_ms"]
                z = compute_z_score(latency, bl["mean"], bl["std"])

                deviations.append(ClinicalDeviation(
                    domain_name="Auditory Response Latency",
                    dsm5_code="A.1 (Response to social cues)",
                    metric_value=latency,
                    baseline_mean=bl["mean"],
                    baseline_std=bl["std"],
                    z_score=z,
                    interpretation=(
                        f"Head-turn response latency: "
                        f"{latency:.0f}ms "
                        f"(baseline: {bl['mean']}+-{bl['std']}ms). "
                        f"Deviation: {z:+.1f} SD. "
                        f"{'Delayed response.' if z > 1 else 'Within typical range.'}"
                    ),
                    # HIGH latency = concerning
                    clinical_significance=interpret_z_score(
                        z, higher_is_worse=True
                    )
                ))
            else:
                # No response at all
                bl = BASELINES["name_call_latency_ms"]
                deviations.append(ClinicalDeviation(
                    domain_name="Auditory Response Latency",
                    dsm5_code="A.1 (Response to social cues)",
                    metric_value=-1,
                    baseline_mean=bl["mean"],
                    baseline_std=bl["std"],
                    z_score=3.5,  # Capped at 3.5 for no response
                    interpretation=(
                        "No head-turn response detected within "
                        "the observation window after auditory "
                        "stimulus. Absence of orienting response "
                        "is a strong indicator per Nadig et al. "
                        "(2007). Note: environmental factors "
                        "(audio volume, ambient noise) may affect "
                        "this metric."
                    ),
                    clinical_significance="atypical"
                ))

        # =================================================
        # DOMAIN 3: EMOTIONAL RECIPROCITY (DSM-5 A.2)
        # Lower smile-back rate = more concerning
        # =================================================
        if stimulus_metrics and stimulus_metrics.get("reciprocity"):
            rc = stimulus_metrics["reciprocity"]
            smile_pct = rc.get("smile_reciprocity_pct", 0)
            bl = BASELINES["smile_reciprocity_pct"]
            z = compute_z_score(smile_pct, bl["mean"], bl["std"])

            # Build rich interpretation
            interp_parts = [
                f"Smile reciprocity rate: {smile_pct:.0f}% "
                f"(baseline: {bl['mean']}+-{bl['std']}%). "
                f"Deviation: {z:+.1f} SD."
            ]

            # Add advanced metrics if available
            ctx_pct = rc.get("contextual_smile_pct", 0)
            if ctx_pct > 0:
                interp_parts.append(
                    f"Contextual smiling: {ctx_pct:.0f}% of "
                    f"total smiles occurred during prompt."
                )

            congruence = rc.get("emotional_congruence", 0)
            if congruence > 0:
                interp_parts.append(
                    f"Emotional congruence: {congruence:.2f} "
                    f"({'High' if congruence > 0.6 else 'Low' if congruence < 0.3 else 'Moderate'} "
                    f"match between stimulus and response)."
                )

            synchrony = rc.get("affect_synchrony", 0)
            if synchrony != 0:
                interp_parts.append(
                    f"Affect synchrony r={synchrony:.2f}."
                )

            diversity = rc.get("expression_diversity", 0)
            if diversity > 0:
                interp_parts.append(
                    f"Expression diversity: {diversity:.2f} "
                    f"({'Rich' if diversity > 0.5 else 'Limited'} range)."
                )

            peak = rc.get("peak_smile_score", 0)
            if peak > 0:
                interp_parts.append(
                    f"Peak smile intensity: {peak:.2f}."
                )

            deviations.append(ClinicalDeviation(
                domain_name="Emotional Reciprocity",
                dsm5_code="A.2 (Nonverbal communication)",
                metric_value=smile_pct,
                baseline_mean=bl["mean"],
                baseline_std=bl["std"],
                z_score=z,
                interpretation=" ".join(interp_parts),
                # LOW reciprocity = concerning
                clinical_significance=interpret_z_score(
                    z, higher_is_worse=False
                )
            ))

        # =================================================
        # DOMAIN 4: GAZE AVOIDANCE (DSM-5 A.2)
        # Higher gaze away percentage = more concerning
        # Lower gaze away = good (typical or better)
        # =================================================
        gaze_pct = face_stats.get("gaze_away_percentage", 0)
        bl = BASELINES["gaze_away_pct"]
        z = compute_z_score(gaze_pct, bl["mean"], bl["std"])

        gaze_events = face_stats.get("gaze_events", [])

        deviations.append(ClinicalDeviation(
            domain_name="Gaze Avoidance",
            dsm5_code="A.2 (Eye contact deficits)",
            metric_value=gaze_pct,
            baseline_mean=bl["mean"],
            baseline_std=bl["std"],
            z_score=z,
            interpretation=(
                f"Gaze away: {gaze_pct:.1f}% of session "
                f"(baseline: {bl['mean']}+-{bl['std']}%). "
                f"Deviation: {z:+.1f} SD. "
                f"Extended gaze-away events (>3s): "
                f"{len(gaze_events)}. "
                f"{'Elevated avoidance pattern.' if z > 1 else ''}"
                f"{'Below average - strong eye contact.' if z < -1 else ''}"
                f"{'Within typical range.' if -1 <= z <= 1 else ''}"
            ),
            # HIGH gaze away = concerning
            clinical_significance=interpret_z_score(
                z, higher_is_worse=True
            )
        ))

        # =================================================
        # DOMAIN 5: FACIAL AFFECT RANGE (DSM-5 A.2)
        # Lower expression variance = more concerning (flat)
        # =================================================
        expr_var = face_stats.get("expression_variance", 0.045)
        bl = BASELINES["expression_variance"]
        z = compute_z_score(expr_var, bl["mean"], bl["std"])

        deviations.append(ClinicalDeviation(
            domain_name="Facial Affect Range",
            dsm5_code="A.2 (Reduced facial expression)",
            metric_value=expr_var,
            baseline_mean=bl["mean"],
            baseline_std=bl["std"],
            z_score=z,
            interpretation=(
                f"Expression variance: {expr_var:.4f} "
                f"(baseline: {bl['mean']}+-{bl['std']}). "
                f"Deviation: {z:+.1f} SD. "
                f"{'Reduced affect range - possible flat affect.' if z < -1 else ''}"
                f"{'Expressive - within or above typical range.' if z >= -1 else ''}"
            ),
            # LOW variance = concerning (flat affect)
            clinical_significance=interpret_z_score(
                z, higher_is_worse=False
            )
        ))

        # =================================================
        # DOMAIN 6: BLINK RATE (Physiological Biomarker)
        # Both too high AND too low are concerning
        # =================================================
        bpm = face_stats.get("blinks_per_minute", 17)
        bl = BASELINES["blinks_per_minute"]
        z = compute_z_score(bpm, bl["mean"], bl["std"])

        blink_direction = ""
        if z < -1:
            blink_direction = (
                "Hypo-blinking may indicate sustained "
                "fixation or sensory processing differences."
            )
        elif z > 1:
            blink_direction = (
                "Hyper-blinking may indicate anxiety "
                "or sensory sensitivity."
            )
        else:
            blink_direction = "Within typical range."

        deviations.append(ClinicalDeviation(
            domain_name="Blink Rate",
            dsm5_code="Physiological biomarker",
            metric_value=bpm,
            baseline_mean=bl["mean"],
            baseline_std=bl["std"],
            z_score=z,
            interpretation=(
                f"Blink rate: {bpm:.1f}/min "
                f"(baseline: {bl['mean']}+-{bl['std']}/min). "
                f"Deviation: {z:+.1f} SD. "
                f"{blink_direction}"
            ),
            # BOTH directions concerning (bilateral)
            clinical_significance=interpret_z_score_bilateral(z)
        ))

        # =================================================
        # DOMAIN 7: MOTOR BEHAVIOR (DSM-5 B.1)
        # Only included if body data is available
        # =================================================
        if (
            body_stats
            and body_stats.get("total_frames_analyzed", 0) > 30
        ):
            rock_pct = body_stats.get("rocking_percentage", 0)
            flap_pct = body_stats.get("flapping_percentage", 0)
            motor_combined = rock_pct * 0.6 + flap_pct * 0.4

            # No published SD baseline for webcam motor detection
            # Use evidence-based categorical thresholds
            if motor_combined > 10:
                motor_sig = "atypical"
                motor_z = 2.5
            elif motor_combined > 3:
                motor_sig = "borderline"
                motor_z = 1.5
            else:
                motor_sig = "typical"
                motor_z = 0.3

            deviations.append(ClinicalDeviation(
                domain_name="Repetitive Motor Behavior",
                dsm5_code="B.1 (Stereotyped movements)",
                metric_value=motor_combined,
                baseline_mean=0.0,
                baseline_std=5.0,
                z_score=motor_z,
                interpretation=(
                    f"Rocking: {rock_pct:.1f}% of frames. "
                    f"Flapping: {flap_pct:.1f}% of frames. "
                    f"Combined motor index: {motor_combined:.1f}. "
                    f"Assessed per DSM-5-TR B.1 criteria."
                ),
                clinical_significance=motor_sig
            ))

        # =================================================
        # STORE DEVIATIONS
        # =================================================
        assessment.deviations = deviations
        assessment.evidence_items = sorted(
            self.evidence, key=lambda e: e.timestamp
        )

        # Domain scores for backward compatibility
        for dev in deviations:
            assessment.domain_scores[dev.domain_name] = round(
                abs(dev.z_score), 1
            )

        # =================================================
        # RECOMMENDATIONS
        # =================================================
        atypical_domains = [
            d for d in deviations
            if d.clinical_significance == "atypical"
        ]
        borderline_domains = [
            d for d in deviations
            if d.clinical_significance == "borderline"
        ]
        typical_domains = [
            d for d in deviations
            if d.clinical_significance == "typical"
        ]

        # General recommendations based on severity
        if atypical_domains:
            domain_names = ", ".join(
                d.domain_name for d in atypical_domains
            )
            assessment.recommendations.append(
                f"Significant deviations (>2 SD) detected in: "
                f"{domain_names}. "
                f"Referral to developmental pediatrician "
                f"recommended for comprehensive evaluation "
                f"using ADOS-2 or equivalent standardized "
                f"instrument."
            )

        if borderline_domains:
            domain_names = ", ".join(
                d.domain_name for d in borderline_domains
            )
            assessment.recommendations.append(
                f"Borderline findings (1-2 SD) in: "
                f"{domain_names}. "
                f"Recommend follow-up screening in 3-6 months "
                f"to track developmental trajectory."
            )

        if not atypical_domains and not borderline_domains:
            assessment.recommendations.append(
                "All measured behavioral domains fall within "
                "neurotypical ranges (<1 SD from baseline). "
                "Continue routine developmental monitoring "
                "per AAP guidelines."
            )

        # Stimulus-specific recommendations
        if stimulus_metrics:
            # Social preference
            sg = stimulus_metrics.get("social_geometric", {})
            social_pct = sg.get("social_preference_pct", 100)
            if social_pct < 40:
                assessment.recommendations.append(
                    f"Low social visual preference "
                    f"({social_pct:.0f}%). "
                    f"This pattern is consistent with findings "
                    f"from Jones & Klin (2013) and the "
                    f"EarliPoint diagnostic platform. "
                    f"Recommend formal eye-tracking assessment."
                )

            # Name-call
            nc = stimulus_metrics.get("name_call", {})
            if not nc.get("responded", True):
                assessment.recommendations.append(
                    "Subject did not orient to auditory stimulus. "
                    "Failure to respond to name is a key early "
                    "indicator per Nadig et al. (2007). "
                    "Note: ensure adequate audio volume in test "
                    "environment. Recommend audiological screening "
                    "to rule out hearing deficit, followed by "
                    "developmental evaluation if hearing is normal."
                )
            elif nc.get("latency_ms", 0) > 1400:
                assessment.recommendations.append(
                    f"Delayed auditory response "
                    f"({nc['latency_ms']:.0f}ms, "
                    f"typical: 500-1100ms). "
                    f"Consider attention and sensory processing "
                    f"evaluation."
                )

            # Reciprocity
            rc = stimulus_metrics.get("reciprocity", {})
            smile_pct = rc.get("smile_reciprocity_pct", 100)
            if smile_pct < 30:
                assessment.recommendations.append(
                    f"Very low emotional reciprocity "
                    f"({smile_pct:.0f}% smile-back rate). "
                    f"Reduced social mirroring is a core feature "
                    f"per Trevisan et al. (2018) meta-analysis. "
                    f"Recommend social-emotional development "
                    f"assessment."
                )

            # Affect synchrony (if available)
            synchrony = rc.get("affect_synchrony", 0)
            if synchrony < -0.1:
                assessment.recommendations.append(
                    f"Negative affect synchrony detected "
                    f"(r={synchrony:.2f}). Subject's expressions "
                    f"inversely correlated with stimulus. "
                    f"This may indicate atypical emotional "
                    f"processing. Consider social cognition "
                    f"evaluation."
                )

        # =================================================
        # OVERALL RISK LEVEL
        # =================================================
        z_scores_for_concern = []
        for d in deviations:
            # Use the "concerning direction" z-score
            if d.clinical_significance == "atypical":
                z_scores_for_concern.append(abs(d.z_score))
            elif d.clinical_significance == "borderline":
                z_scores_for_concern.append(abs(d.z_score))
            else:
                z_scores_for_concern.append(0.0)

        if z_scores_for_concern:
            max_z = max(abs(d.z_score) for d in deviations)
            mean_concern = float(np.mean(z_scores_for_concern))
            n_atypical = len(atypical_domains)

            # Map to 0-100 for backward compatibility
            assessment.overall_risk_score = round(
                min(mean_concern * 25, 100), 1
            )

            # Risk level based on count and severity
            if n_atypical >= 3 or max_z >= 3.5:
                assessment.risk_level = "High"
            elif n_atypical >= 1 or max_z >= 2.5:
                assessment.risk_level = "Elevated"
            elif len(borderline_domains) >= 2:
                assessment.risk_level = "Borderline"
            elif len(borderline_domains) >= 1:
                assessment.risk_level = "Low-Borderline"
            else:
                assessment.risk_level = "Typical"

        # =================================================
        # SUMMARY
        # =================================================
        duration = face_stats.get("session_duration_seconds", 0)
        n_atypical = len(atypical_domains)
        n_borderline = len(borderline_domains)
        n_typical = len(typical_domains)

        # Build domain summary
        parts = []
        if n_atypical > 0:
            parts.append(f"{n_atypical} atypical (>2 SD)")
        if n_borderline > 0:
            parts.append(f"{n_borderline} borderline (1-2 SD)")
        if n_typical > 0:
            parts.append(f"{n_typical} typical (<1 SD)")
        domain_summary = ", ".join(parts) if parts else "No data"

        assessment.summary = (
            f"Structured screening protocol analyzed "
            f"{duration:.0f} seconds of behavioral data across "
            f"{len(deviations)} clinical domains. "
            f"Results: {domain_summary}. "
            f"{len(assessment.evidence_items)} evidence items "
            f"were flagged during the session. "
        )

        # Add specific atypical findings to summary
        if n_atypical > 0:
            atypical_details = [
                f"{d.domain_name} (z={d.z_score:+.1f})"
                for d in atypical_domains
            ]
            assessment.summary += (
                f"Significant deviations identified in: "
                f"{'; '.join(atypical_details)}. "
            )

        # Add note about directionality
        if n_atypical == 0 and n_borderline == 0:
            assessment.summary += (
                "No clinically significant deviations were "
                "detected. All behavioral metrics fall within "
                "expected neurotypical ranges. "
            )

        assessment.summary += (
            "All metrics are compared against published "
            "neurotypical baselines with proper directionality "
            "(e.g., low gaze avoidance is typical, not atypical). "
            "This is a screening tool only and does NOT "
            "constitute a clinical diagnosis. Results must be "
            "reviewed by a qualified healthcare professional."
        )

        return assessment

    # =============================================
    # RESET
    # =============================================

    def reset(self):
        """Reset all evidence for a new session."""
        self.evidence = []
        self.gaze_away_frames = 0
        self.total_face_frames = 0
        self.flat_affect_frames = 0
        self.session_start = time.time()
        self._last_gaze_flag = 0.0
        self._last_expression_flag = 0.0
        self._last_motor_flag = 0.0

        # Clear evidence screenshots
        for f in EVIDENCE_DIR.glob("evidence_*.jpg"):
            try:
                f.unlink()
            except Exception:
                pass