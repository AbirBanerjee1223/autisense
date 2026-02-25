# modules/reciprocity_tracker.py

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque


@dataclass
class SmileEvent:
    """A single detected smile episode."""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    peak_intensity: float = 0.0
    avg_intensity: float = 0.0
    was_contextual: bool = False    # During prompt = contextual
    latency_from_prompt_ms: float = -1.0


@dataclass
class MirroringEvent:
    """A detected emotional mirroring response."""
    timestamp: float = 0.0
    session_time: float = 0.0
    expression_type: str = ""     # smile, brow_raise, surprise
    intensity: float = 0.0
    latency_ms: float = 0.0       # Time from stimulus change
    is_congruent: bool = False     # Matches the stimulus emotion


@dataclass
class ReciprocityResult:
    """Frame-by-frame reciprocity analysis."""
    is_smiling: bool = False
    smile_intensity: float = 0.0
    is_mirroring: bool = False
    current_expression: str = "neutral"
    expression_congruence: float = 0.0  # 0-1, how well they match stimulus


@dataclass
class ReciprocityReport:
    """Complete reciprocity analysis for the session."""
    # Core metrics
    smile_reciprocity_pct: float = 0.0
    contextual_smile_pct: float = 0.0
    non_contextual_smile_pct: float = 0.0
    mean_smile_latency_ms: float = -1.0
    mean_smile_duration_ms: float = 0.0
    peak_smile_intensity: float = 0.0

    # Advanced metrics
    emotional_congruence_score: float = 0.0
    mirroring_frequency: float = 0.0      # Events per minute
    affect_synchrony_score: float = 0.0   # Correlation of stimulus vs response
    expression_diversity: float = 0.0     # Range of expressions shown

    # Events
    smile_events: List[SmileEvent] = field(default_factory=list)
    mirroring_events: List[MirroringEvent] = field(default_factory=list)

    # Time series (for visualization)
    stimulus_intensity_curve: List[float] = field(default_factory=list)
    response_intensity_curve: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Counts
    total_frames: int = 0
    smile_frames: int = 0
    prompted_frames: int = 0    # Frames during active prompt
    smile_during_prompt: int = 0


class ReciprocityTracker:
    """
    Advanced emotional reciprocity tracking system.

    Measures:
    1. Smile reciprocity (does subject smile when prompted?)
    2. Smile latency (how fast do they respond?)
    3. Contextual vs non-contextual smiling
    4. Emotional congruence (do their expressions match?)
    5. Affect synchrony (correlation between stimulus and response)
    6. Expression diversity (range of expressions shown)
    7. Mirroring events (discrete emotional matching moments)

    Clinical basis:
    - Trevisan et al. (2018): Reduced facial expression
      production in autism
    - Stel et al. (2008): Reduced automatic mimicry in ASD
    - McIntosh et al. (2006): Voluntary vs spontaneous
      facial mimicry differences in ASD
    """

    # Expression thresholds
    SMILE_THRESHOLD = 0.3
    BROW_RAISE_THRESHOLD = 0.25
    SURPRISE_THRESHOLD = 0.35  # jaw_open + brow_raise
    EXPRESSION_CHANGE_THRESHOLD = 0.15

    def __init__(self):
        self.session_start: float = 0.0
        self.prompt_active: bool = False
        self.prompt_start_time: float = 0.0

        # Smile episode tracking
        self._in_smile: bool = False
        self._smile_start: float = 0.0
        self._smile_intensities: List[float] = []
        self._smile_events: List[SmileEvent] = []

        # Mirroring events
        self._mirroring_events: List[MirroringEvent] = []

        # Frame-level tracking
        self._total_frames: int = 0
        self._smile_frames: int = 0
        self._prompted_frames: int = 0
        self._smile_during_prompt: int = 0

        # Time series for synchrony analysis
        self._stimulus_curve: List[float] = []
        self._response_curve: List[float] = []
        self._timestamps: List[float] = []

        # Expression history for diversity
        self._expression_history: deque = deque(maxlen=300)
        self._expression_intensity_history: deque = deque(maxlen=300)

        # Previous frame state (for detecting changes)
        self._prev_smile_score: float = 0.0
        self._prev_brow_score: float = 0.0
        self._prev_expression: str = "neutral"

        # Latency tracking
        self._last_stimulus_change_time: float = 0.0
        self._stimulus_smile_active: bool = False
        self._first_smile_detected: bool = False
        self._first_smile_latency_ms: float = -1.0

    def start_session(self):
        """Begin tracking."""
        self.session_start = time.time()

    def set_prompt_active(self, active: bool, stimulus_smile_intensity: float = 0.0):
        """
        Called by the stimulus engine to indicate whether
        a smile prompt is currently being displayed.

        Args:
            active: Whether smile prompt is showing
            stimulus_smile_intensity: How much the stimulus
                                     is smiling (0-1)
        """
        now = time.time()

        if active and not self.prompt_active:
            # Prompt just started
            self.prompt_start_time = now
            self._last_stimulus_change_time = now

        # Track stimulus changes for mirroring detection
        if active and stimulus_smile_intensity > 0.3:
            if not self._stimulus_smile_active:
                self._stimulus_smile_active = True
                self._last_stimulus_change_time = now
        elif self._stimulus_smile_active:
            self._stimulus_smile_active = False

        self.prompt_active = active

    def process_frame(
        self,
        smile_score: float,
        brow_raise_score: float,
        jaw_open_score: float,
        expression_label: str,
        stimulus_intensity: float = 0.0,
    ) -> ReciprocityResult:
        """
        Process a single frame of expression data.

        Args:
            smile_score: 0-1 smile intensity from blendshapes
            brow_raise_score: 0-1 brow raise from blendshapes
            jaw_open_score: 0-1 jaw openness from blendshapes
            expression_label: Current expression label
            stimulus_intensity: How intense the stimulus 
                               expression is (0-1)

        Returns:
            ReciprocityResult for this frame
        """
        now = time.time()
        self._total_frames += 1

        result = ReciprocityResult()
        result.current_expression = expression_label
        result.smile_intensity = smile_score

        # ===== SMILE DETECTION =====
        is_smiling = smile_score > self.SMILE_THRESHOLD
        result.is_smiling = is_smiling

        if is_smiling:
            self._smile_frames += 1

        # Track during prompt
        if self.prompt_active:
            self._prompted_frames += 1
            if is_smiling:
                self._smile_during_prompt += 1

        # ===== SMILE EPISODE TRACKING =====
        if is_smiling and not self._in_smile:
            # Smile onset
            self._in_smile = True
            self._smile_start = now
            self._smile_intensities = [smile_score]

            # First smile latency
            if self.prompt_active and not self._first_smile_detected:
                self._first_smile_detected = True
                self._first_smile_latency_ms = (
                    (now - self.prompt_start_time) * 1000
                )

        elif is_smiling and self._in_smile:
            # Continuing smile
            self._smile_intensities.append(smile_score)

        elif not is_smiling and self._in_smile:
            # Smile offset — record the episode
            self._in_smile = False
            duration_ms = (now - self._smile_start) * 1000

            event = SmileEvent(
                start_time=self._smile_start,
                end_time=now,
                duration_ms=duration_ms,
                peak_intensity=max(self._smile_intensities),
                avg_intensity=np.mean(self._smile_intensities),
                was_contextual=self.prompt_active,
                latency_from_prompt_ms=(
                    (self._smile_start - self.prompt_start_time) * 1000
                    if self.prompt_active else -1.0
                )
            )
            self._smile_events.append(event)

        # ===== MIRRORING DETECTION =====
        # Check if subject's expression change matches stimulus
        expression_changed = (
            abs(smile_score - self._prev_smile_score)
            > self.EXPRESSION_CHANGE_THRESHOLD
        )

        if expression_changed and self.prompt_active:
            # Calculate latency from last stimulus change
            latency = (now - self._last_stimulus_change_time) * 1000

            # Determine if the response is congruent
            # (subject smiles when stimulus smiles)
            is_congruent = (
                (smile_score > self.SMILE_THRESHOLD and
                 stimulus_intensity > 0.3)
                or
                (smile_score < self.SMILE_THRESHOLD and
                 stimulus_intensity < 0.3)
            )

            if latency < 5000:  # Only count if within 5 seconds
                event = MirroringEvent(
                    timestamp=now,
                    session_time=now - self.session_start,
                    expression_type=(
                        "smile" if smile_score > self.SMILE_THRESHOLD
                        else "neutral_return"
                    ),
                    intensity=smile_score,
                    latency_ms=latency,
                    is_congruent=is_congruent
                )
                self._mirroring_events.append(event)

            result.is_mirroring = is_congruent

        # ===== CONGRUENCE SCORE =====
        # How well does current expression match what's expected?
        if self.prompt_active and stimulus_intensity > 0:
            # During active prompt: congruence = similarity
            result.expression_congruence = 1.0 - abs(
                stimulus_intensity - smile_score
            )
        else:
            result.expression_congruence = 0.5  # Neutral baseline

        # ===== TIME SERIES =====
        self._stimulus_curve.append(stimulus_intensity)
        self._response_curve.append(smile_score)
        self._timestamps.append(now - self.session_start)

        # ===== EXPRESSION DIVERSITY =====
        self._expression_history.append(expression_label)
        combined_intensity = (
            smile_score * 0.4 +
            brow_raise_score * 0.3 +
            jaw_open_score * 0.3
        )
        self._expression_intensity_history.append(combined_intensity)

        # Update previous state
        self._prev_smile_score = smile_score
        self._prev_brow_score = brow_raise_score
        self._prev_expression = expression_label

        return result

    def compute_report(self) -> ReciprocityReport:
        """
        Compute the complete reciprocity analysis report.
        Called at the end of the session.
        """
        report = ReciprocityReport()

        # ===== BASIC COUNTS =====
        report.total_frames = self._total_frames
        report.smile_frames = self._smile_frames
        report.prompted_frames = self._prompted_frames
        report.smile_during_prompt = self._smile_during_prompt
        report.smile_events = self._smile_events
        report.mirroring_events = self._mirroring_events

        # ===== SMILE RECIPROCITY =====
        if self._prompted_frames > 0:
            report.smile_reciprocity_pct = (
                self._smile_during_prompt /
                self._prompted_frames * 100
            )
        else:
            report.smile_reciprocity_pct = 0.0

        # ===== CONTEXTUAL VS NON-CONTEXTUAL =====
        contextual_smiles = [
            e for e in self._smile_events if e.was_contextual
        ]
        non_contextual_smiles = [
            e for e in self._smile_events if not e.was_contextual
        ]

        total_smile_time = sum(
            e.duration_ms for e in self._smile_events
        )

        if total_smile_time > 0:
            contextual_time = sum(
                e.duration_ms for e in contextual_smiles
            )
            report.contextual_smile_pct = (
                contextual_time / total_smile_time * 100
            )
            report.non_contextual_smile_pct = (
                100 - report.contextual_smile_pct
            )

        # ===== LATENCY =====
        report.mean_smile_latency_ms = self._first_smile_latency_ms

        contextual_latencies = [
            e.latency_from_prompt_ms
            for e in contextual_smiles
            if e.latency_from_prompt_ms >= 0
        ]
        if contextual_latencies:
            report.mean_smile_latency_ms = float(
                np.mean(contextual_latencies)
            )

        # ===== DURATION =====
        if self._smile_events:
            report.mean_smile_duration_ms = float(
                np.mean([e.duration_ms for e in self._smile_events])
            )
            report.peak_smile_intensity = float(
                max(e.peak_intensity for e in self._smile_events)
            )

        # ===== EMOTIONAL CONGRUENCE =====
        if self._mirroring_events:
            congruent = [
                e for e in self._mirroring_events if e.is_congruent
            ]
            report.emotional_congruence_score = (
                len(congruent) / len(self._mirroring_events)
            )

        # ===== MIRRORING FREQUENCY =====
        elapsed = (
            time.time() - self.session_start
            if self.session_start > 0 else 1.0
        )
        if elapsed > 0:
            report.mirroring_frequency = (
                len(self._mirroring_events) / (elapsed / 60.0)
            )

        # ===== AFFECT SYNCHRONY =====
        # Pearson correlation between stimulus and response curves
        if (
            len(self._stimulus_curve) > 10
            and len(self._response_curve) > 10
        ):
            stim_arr = np.array(self._stimulus_curve)
            resp_arr = np.array(self._response_curve)

            # Only compute during prompted period
            if np.std(stim_arr) > 0.01 and np.std(resp_arr) > 0.01:
                correlation = np.corrcoef(stim_arr, resp_arr)[0, 1]
                if not np.isnan(correlation):
                    report.affect_synchrony_score = float(correlation)

        # ===== EXPRESSION DIVERSITY =====
        if self._expression_history:
            unique_expressions = len(
                set(self._expression_history)
            )
            # Normalize: 1 expression = 0, 5+ = 1.0
            report.expression_diversity = min(
                (unique_expressions - 1) / 4.0, 1.0
            )

        # ===== TIME SERIES =====
        report.stimulus_intensity_curve = self._stimulus_curve.copy()
        report.response_intensity_curve = self._response_curve.copy()
        report.timestamps = self._timestamps.copy()

        return report

    def get_live_metrics(self) -> dict:
        """
        Get current metrics for live display during screening.
        Lightweight version of compute_report().
        """
        reciprocity_pct = 0.0
        if self._prompted_frames > 0:
            reciprocity_pct = (
                self._smile_during_prompt /
                self._prompted_frames * 100
            )

        peak_intensity = 0.0
        if self._smile_events:
            peak_intensity = max(
                e.peak_intensity for e in self._smile_events
            )

        return {
            "smile_reciprocity_pct": round(reciprocity_pct, 1),
            "smile_episodes": len(self._smile_events),
            "contextual_smiles": len(
                [e for e in self._smile_events if e.was_contextual]
            ),
            "first_smile_latency_ms": round(
                self._first_smile_latency_ms, 0
            ),
            "peak_intensity": round(peak_intensity, 3),
            "mirroring_events": len(self._mirroring_events),
            "is_currently_smiling": self._in_smile,
            "current_smile_duration_ms": (
                (time.time() - self._smile_start) * 1000
                if self._in_smile else 0
            ),
        }

    def reset(self):
        """Reset all tracking."""
        self.prompt_active = False
        self.prompt_start_time = 0.0
        self._in_smile = False
        self._smile_start = 0.0
        self._smile_intensities = []
        self._smile_events = []
        self._mirroring_events = []
        self._total_frames = 0
        self._smile_frames = 0
        self._prompted_frames = 0
        self._smile_during_prompt = 0
        self._stimulus_curve = []
        self._response_curve = []
        self._timestamps = []
        self._expression_history.clear()
        self._expression_intensity_history.clear()
        self._prev_smile_score = 0.0
        self._prev_brow_score = 0.0
        self._prev_expression = "neutral"
        self._first_smile_detected = False
        self._first_smile_latency_ms = -1.0
        self.session_start = time.time()