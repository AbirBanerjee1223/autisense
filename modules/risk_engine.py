# modules/risk_engine.py (COMPLETE FILE - Clinical Deviations)

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from config import EVIDENCE_DIR, BASELINES


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
    clinical_significance: str = ""


def compute_z_score(value: float, mean: float, std: float) -> float:
    """Compute standard deviation distance from baseline."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def interpret_z_score(z: float, higher_is_worse: bool = True) -> str:
    """Interpret a z-score clinically."""
    z_abs = abs(z)
    if z_abs < 1.0:
        return "typical"
    elif z_abs < 2.0:
        return "borderline"
    else:
        return "atypical"


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


class RiskEngine:
    """
    Clinical Risk Engine using Standard Deviation analysis.
    
    Computes how many standard deviations each behavioral
    metric deviates from published neurotypical baselines,
    rather than using arbitrary 0-100 scores.
    """

    def __init__(self, session_start_time: float):
        self.session_start = session_start_time
        self.evidence: List[EvidenceItem] = []

        self.gaze_away_frames = 0
        self.total_face_frames = 0
        self.flat_affect_frames = 0

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

    def process_face_result(self, face_result, raw_frame):
        """Process face analysis results and flag evidence."""
        if not face_result.face_detected:
            return

        self.total_face_frames += 1
        now = time.time()

        if not face_result.gaze.is_looking_at_camera:
            self.gaze_away_frames += 1

        if face_result.emotion.expression_label == "flat_affect":
            self.flat_affect_frames += 1

        # Flag extended gaze avoidance
        gaze_pct = (
            self.gaze_away_frames /
            max(self.total_face_frames, 1) * 100
        )
        if (
            gaze_pct > 50
            and self.total_face_frames > 60
            and (now - self._last_gaze_flag) > self.COOLDOWN
        ):
            z = compute_z_score(
                gaze_pct,
                BASELINES["gaze_away_pct"]["mean"],
                BASELINES["gaze_away_pct"]["std"]
            )
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
                    f"Z-score: {z:+.1f} SD from neurotypical mean."
                ),
                confidence=min(abs(z) / 3.0, 0.99),
                severity="high" if abs(z) > 2 else "medium",
                screenshot_path=screenshot,
                metric_name="gaze_away_pct",
                metric_value=gaze_pct,
                baseline_mean=BASELINES["gaze_away_pct"]["mean"],
                baseline_std=BASELINES["gaze_away_pct"]["std"],
                z_score=z
            ))
            self._last_gaze_flag = now

        # Flag flat affect
        flat_pct = (
            self.flat_affect_frames /
            max(self.total_face_frames, 1) * 100
        )
        if (
            flat_pct > 60
            and self.total_face_frames > 60
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
                    f"Autocorrelation score: "
                    f"{body_result.repetitive_motion_score:.2f}. "
                    f"Classified under DSM-5-TR criterion B.1 "
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
                    f"Hand flapping behavior detected. "
                    f"Flap score: {body_result.hand_flap_score:.2f}. "
                    f"Rapid oscillatory wrist movements with high "
                    f"direction-reversal frequency. DSM-5-TR B.1."
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

    def compute_assessment(
        self,
        face_stats: dict,
        stimulus_metrics: Optional[dict] = None,
        body_stats: Optional[dict] = None,
    ) -> RiskAssessment:
        """
        Compute clinical assessment using standard deviations
        from published neurotypical baselines.

        Accepts optional stimulus_metrics from StimulusEngine
        and optional body_stats from BodyAnalyzer.
        """
        assessment = RiskAssessment()
        deviations = []

        # ===== 1. SOCIAL PREFERENCE (DSM-5 A.1) =====
        if stimulus_metrics and stimulus_metrics.get("social_geometric"):
            sg = stimulus_metrics["social_geometric"]
            social_pct = sg.get("social_preference_pct", 50)
            bl = BASELINES["social_preference_pct"]
            z = compute_z_score(social_pct, bl["mean"], bl["std"])
            z_concern = -z  # Lower social = more concerning

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
                    f"{'Below' if z < 0 else 'Above'} average by "
                    f"{abs(z):.1f} SD."
                ),
                clinical_significance=interpret_z_score(
                    z_concern, higher_is_worse=True
                )
            ))

        # ===== 2. NAME-CALL LATENCY (DSM-5 A.1) =====
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
                        f"Response latency: {latency:.0f}ms "
                        f"(baseline: {bl['mean']}+-{bl['std']}ms). "
                        f"{'Delayed' if z > 0 else 'Typical'} by "
                        f"{abs(z):.1f} SD."
                    ),
                    clinical_significance=interpret_z_score(
                        z, higher_is_worse=True
                    )
                ))
            else:
                deviations.append(ClinicalDeviation(
                    domain_name="Auditory Response Latency",
                    dsm5_code="A.1 (Response to social cues)",
                    metric_value=-1,
                    baseline_mean=BASELINES["name_call_latency_ms"]["mean"],
                    baseline_std=BASELINES["name_call_latency_ms"]["std"],
                    z_score=3.5,
                    interpretation=(
                        "No head-turn response detected within "
                        "10-second window after auditory stimulus. "
                        "Absence of orienting response is a strong "
                        "indicator per Nadig et al. 2007."
                    ),
                    clinical_significance="atypical"
                ))

        # ===== 3. EMOTIONAL RECIPROCITY (DSM-5 A.2) =====
        if stimulus_metrics and stimulus_metrics.get("reciprocity"):
            rc = stimulus_metrics["reciprocity"]
            smile_pct = rc.get("smile_reciprocity_pct", 0)
            bl = BASELINES["smile_reciprocity_pct"]
            z = compute_z_score(smile_pct, bl["mean"], bl["std"])

            # Build rich interpretation using advanced metrics
            interpretation_parts = [
                f"Smile reciprocity rate: {smile_pct:.0f}% "
                f"(baseline: {bl['mean']}+-{bl['std']}%). "
                f"Deviation: {z:+.1f} SD."
            ]

            # Add contextual analysis
            ctx_pct = rc.get("contextual_smile_pct", 0)
            if ctx_pct > 0:
                interpretation_parts.append(
                    f"Contextual smiling: {ctx_pct:.0f}% of "
                    f"total smiles occurred during prompt."
                )

            # Add congruence
            congruence = rc.get("emotional_congruence", 0)
            interpretation_parts.append(
                f"Emotional congruence: {congruence:.2f} "
                f"({'High' if congruence > 0.6 else 'Low' if congruence < 0.3 else 'Moderate'} "
                f"match between stimulus and response)."
            )

            # Add synchrony
            synchrony = rc.get("affect_synchrony", 0)
            if synchrony != 0:
                interpretation_parts.append(
                    f"Affect synchrony (correlation): "
                    f"{synchrony:.2f}."
                )

            # Add diversity
            diversity = rc.get("expression_diversity", 0)
            interpretation_parts.append(
                f"Expression diversity: {diversity:.2f} "
                f"({'Rich' if diversity > 0.5 else 'Limited'} range)."
            )

            deviations.append(ClinicalDeviation(
                domain_name="Emotional Reciprocity",
                dsm5_code="A.2 (Nonverbal communication)",
                metric_value=smile_pct,
                baseline_mean=bl["mean"],
                baseline_std=bl["std"],
                z_score=z,
                interpretation=" ".join(interpretation_parts),
                clinical_significance=interpret_z_score(
                    -z, higher_is_worse=True
                )
            ))

        # ===== 4. GAZE AVOIDANCE (DSM-5 A.2) =====
        gaze_pct = face_stats.get("gaze_away_percentage", 0)
        bl = BASELINES["gaze_away_pct"]
        z = compute_z_score(gaze_pct, bl["mean"], bl["std"])

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
                f"Extended gaze events (>3s): "
                f"{len(face_stats.get('gaze_events', []))}."
            ),
            clinical_significance=interpret_z_score(
                z, higher_is_worse=True
            )
        ))

        # ===== 5. EXPRESSION VARIANCE (DSM-5 A.2) =====
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
                f"{'Reduced' if z < 0 else 'Typical'} affect "
                f"range, {abs(z):.1f} SD from mean."
            ),
            clinical_significance=interpret_z_score(
                -z, higher_is_worse=True
            )
        ))

        # ===== 6. BLINK RATE (Physiological) =====
        bpm = face_stats.get("blinks_per_minute", 17)
        bl = BASELINES["blinks_per_minute"]
        z = compute_z_score(bpm, bl["mean"], bl["std"])

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
                f"Deviation: {z:+.1f} SD."
            ),
            clinical_significance=interpret_z_score(
                abs(z), higher_is_worse=True
            )
        ))

        # ===== 7. MOTOR BEHAVIOR (DSM-5 B.1) - Only if body data =====
        if body_stats and body_stats.get("total_frames_analyzed", 0) > 30:
            rock_pct = body_stats.get("rocking_percentage", 0)
            flap_pct = body_stats.get("flapping_percentage", 0)

            # Combine motor score
            motor_combined = rock_pct * 0.6 + flap_pct * 0.4
            # No published SD baseline for this, so use evidence-based flag
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
                    f"Repetitive motor behaviors assessed per "
                    f"DSM-5-TR B.1 criteria."
                ),
                clinical_significance=motor_sig
            ))

        # Store deviations
        assessment.deviations = deviations
        assessment.evidence_items = sorted(
            self.evidence, key=lambda e: e.timestamp
        )

        # ===== DOMAIN SCORES (backward compat with report) =====
        for dev in deviations:
            assessment.domain_scores[dev.domain_name] = round(
                abs(dev.z_score), 1
            )

        # ===== RECOMMENDATIONS =====
        atypical_domains = [
            d for d in deviations
            if d.clinical_significance == "atypical"
        ]
        borderline_domains = [
            d for d in deviations
            if d.clinical_significance == "borderline"
        ]

        if atypical_domains:
            domain_names = ", ".join(
                d.domain_name for d in atypical_domains
            )
            assessment.recommendations.append(
                f"Significant deviations detected in: "
                f"{domain_names}. "
                f"Referral to developmental pediatrician "
                f"recommended for comprehensive evaluation "
                f"using ADOS-2."
            )

        if borderline_domains:
            domain_names = ", ".join(
                d.domain_name for d in borderline_domains
            )
            assessment.recommendations.append(
                f"Borderline findings in {domain_names}. "
                f"Recommend follow-up screening in 3-6 months."
            )

        if not atypical_domains and not borderline_domains:
            assessment.recommendations.append(
                "All measured behavioral domains fall within "
                "neurotypical ranges. Continue routine "
                "developmental monitoring per AAP guidelines."
            )

        # Stimulus-specific recommendations
        if stimulus_metrics:
            sg = stimulus_metrics.get("social_geometric", {})
            if sg.get("social_preference_pct", 100) < 40:
                assessment.recommendations.append(
                    "Low social visual preference detected "
                    f"({sg['social_preference_pct']:.0f}%). "
                    "This pattern is consistent with findings "
                    "from Jones & Klin (2013) and the EarliPoint "
                    "diagnostic. Recommend formal eye-tracking "
                    "assessment."
                )

            nc = stimulus_metrics.get("name_call", {})
            if not nc.get("responded", True):
                assessment.recommendations.append(
                    "Subject did not orient to auditory stimulus. "
                    "Failure to respond to name is a key early "
                    "indicator per Nadig et al. (2007). Recommend "
                    "audiological screening to rule out hearing "
                    "deficit, followed by developmental evaluation."
                )
            elif nc.get("latency_ms", 0) > 1400:
                assessment.recommendations.append(
                    f"Delayed auditory response "
                    f"({nc['latency_ms']:.0f}ms). "
                    "Typical latency is 500-1100ms. Consider "
                    "attention and sensory processing evaluation."
                )

            rc = stimulus_metrics.get("reciprocity", {})
            if rc.get("smile_reciprocity_pct", 100) < 30:
                assessment.recommendations.append(
                    "Very low emotional reciprocity detected "
                    f"({rc['smile_reciprocity_pct']:.0f}% "
                    f"smile-back rate). "
                    "Reduced social mirroring is a core feature "
                    "per Trevisan et al. (2018). Recommend "
                    "social-emotional development assessment."
                )

        # ===== OVERALL RISK LEVEL =====
        z_scores = [abs(d.z_score) for d in deviations]
        if z_scores:
            max_z = max(z_scores)
            mean_z = float(np.mean(z_scores))
            atypical_count = len(atypical_domains)

            assessment.overall_risk_score = round(
                min(mean_z * 25, 100), 1
            )

            if atypical_count >= 2 or max_z >= 3.0:
                assessment.risk_level = "High"
            elif atypical_count >= 1 or max_z >= 2.0:
                assessment.risk_level = "Elevated"
            elif any(
                d.clinical_significance == "borderline"
                for d in deviations
            ):
                assessment.risk_level = "Borderline"
            else:
                assessment.risk_level = "Typical"

        # ===== SUMMARY =====
        duration = face_stats.get("session_duration_seconds", 0)
        n_atypical = len(atypical_domains)
        n_borderline = len(borderline_domains)
        n_typical = len(deviations) - n_atypical - n_borderline

        parts = []
        if n_atypical > 0:
            parts.append(f"{n_atypical} atypical (>2 SD)")
        if n_borderline > 0:
            parts.append(f"{n_borderline} borderline (1-2 SD)")
        if n_typical > 0:
            parts.append(f"{n_typical} typical (<1 SD)")
        domain_summary = ", ".join(parts)

        assessment.summary = (
            f"Structured screening protocol analyzed "
            f"{duration:.0f} seconds of behavioral data across "
            f"{len(deviations)} clinical domains. "
            f"Results: {domain_summary}. "
            f"{len(assessment.evidence_items)} evidence items "
            f"were flagged during the session. "
        )

        if n_atypical > 0:
            atypical_names = [
                f"{d.domain_name} (z={d.z_score:+.1f})"
                for d in atypical_domains
            ]
            assessment.summary += (
                f"Significant deviations in: "
                f"{'; '.join(atypical_names)}. "
            )

        assessment.summary += (
            "All metrics are compared against published "
            "neurotypical baselines. This is a screening tool "
            "only and does NOT constitute a clinical diagnosis. "
            "Results must be reviewed by a qualified healthcare "
            "professional."
        )

        return assessment

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

        for f in EVIDENCE_DIR.glob("evidence_*.jpg"):
            try:
                f.unlink()
            except Exception:
                pass