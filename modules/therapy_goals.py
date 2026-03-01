# modules/therapy_goals.py
"""
Therapy Goal Tracker.
Implements ABA-style goal tracking with discrete trial data collection,
progress monitoring, and phase-change detection.

Modeled on professional ABA data systems: CentralReach, Catalyst, Hi Rasmus.

Tracks:
  - Mastery criteria (e.g., 80% across 3 consecutive sessions)
  - Trial-by-trial data
  - Phase progression (Baseline → Acquisition → Maintenance → Generalization)
  - Progress trends over time
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from modules.data_store import load_json, save_json, get_subject_file


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class TrialEntry:
    """A single data collection entry for a goal."""
    date: str
    successful_trials: int
    total_trials: int
    prompt_level: str           # "independent", "verbal", "gestural", "model", "physical", "full_physical"
    notes: str = ""
    environment: str = ""       # "clinic", "home", "school", "community"

    @property
    def percentage(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return (self.successful_trials / self.total_trials) * 100


@dataclass
class TherapyGoal:
    """A single therapy goal with full tracking data."""
    goal_id: str
    goal_text: str
    domain: str                 # "communication", "social", "behavior", "adaptive", "motor", "academic"
    target_behavior: str
    mastery_criteria: str       # e.g., "80% across 3 consecutive sessions"
    mastery_percentage: float   # e.g., 80.0
    mastery_consecutive: int    # e.g., 3
    phase: str                  # "baseline", "acquisition", "maintenance", "generalization"
    baseline_value: str
    target_value: str
    start_date: str
    trial_data: List[TrialEntry] = field(default_factory=list)
    status: str = "active"      # "active", "mastered", "on_hold", "discontinued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def latest_percentage(self) -> float:
        if not self.trial_data:
            return 0.0
        return self.trial_data[-1].percentage

    @property
    def session_count(self) -> int:
        return len(self.trial_data)

    @property
    def is_mastered(self) -> bool:
        if len(self.trial_data) < self.mastery_consecutive:
            return False
        recent = self.trial_data[-self.mastery_consecutive:]
        return all(t.percentage >= self.mastery_percentage for t in recent)

    @property
    def trend(self) -> str:
        """Calculate trend direction from last 5 data points."""
        if len(self.trial_data) < 3:
            return "insufficient_data"
        recent = [t.percentage for t in self.trial_data[-5:]]
        if len(recent) < 3:
            return "insufficient_data"

        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        diff = second_half - first_half
        if diff > 10:
            return "improving"
        elif diff < -10:
            return "declining"
        else:
            return "stable"

    @property
    def days_active(self) -> int:
        try:
            start = datetime.fromisoformat(self.start_date)
            return (datetime.now() - start).days
        except (ValueError, TypeError):
            return 0

    def get_progress_data(self) -> List[Dict]:
        """Return trial data formatted for charting."""
        return [
            {
                "date": t.date,
                "percentage": t.percentage,
                "successful": t.successful_trials,
                "total": t.total_trials,
                "prompt": t.prompt_level,
                "notes": t.notes,
            }
            for t in self.trial_data
        ]


# ─────────────────────────────────────────────
# PROMPT LEVEL HIERARCHY
# ─────────────────────────────────────────────
PROMPT_LEVELS = {
    "independent": {
        "label": "Independent",
        "icon": "🌟",
        "description": "No prompt needed — child performs skill independently",
        "level": 0,
    },
    "verbal": {
        "label": "Verbal Prompt",
        "icon": "🗣️",
        "description": "Verbal cue or instruction provided",
        "level": 1,
    },
    "gestural": {
        "label": "Gestural Prompt",
        "icon": "👉",
        "description": "Pointing, nodding, or gesture provided",
        "level": 2,
    },
    "model": {
        "label": "Model Prompt",
        "icon": "🪞",
        "description": "Therapist demonstrates the behavior",
        "level": 3,
    },
    "physical": {
        "label": "Partial Physical",
        "icon": "🤝",
        "description": "Light physical guidance (e.g., tap on hand)",
        "level": 4,
    },
    "full_physical": {
        "label": "Full Physical",
        "icon": "🫱",
        "description": "Hand-over-hand assistance",
        "level": 5,
    },
}

PHASE_INFO = {
    "baseline": {
        "label": "Baseline",
        "icon": "📊",
        "color": "#95a5a6",
        "description": "Measuring current skill level before intervention",
    },
    "acquisition": {
        "label": "Acquisition",
        "icon": "📈",
        "color": "#3498db",
        "description": "Actively teaching the skill with prompting",
    },
    "maintenance": {
        "label": "Maintenance",
        "icon": "✅",
        "color": "#2ecc71",
        "description": "Skill mastered — monitoring to ensure retention",
    },
    "generalization": {
        "label": "Generalization",
        "icon": "🌍",
        "color": "#9b59b6",
        "description": "Practicing skill across settings, people, and materials",
    },
}

GOAL_DOMAINS = {
    "communication": {
        "label": "Communication",
        "icon": "💬",
        "color": "#3498db",
    },
    "social": {
        "label": "Social Skills",
        "icon": "👫",
        "color": "#e74c3c",
    },
    "behavior": {
        "label": "Behavior",
        "icon": "🧠",
        "color": "#f39c12",
    },
    "adaptive": {
        "label": "Adaptive / Self-Care",
        "icon": "🏠",
        "color": "#2ecc71",
    },
    "motor": {
        "label": "Motor Skills",
        "icon": "🏃",
        "color": "#1abc9c",
    },
    "academic": {
        "label": "Academic / Cognitive",
        "icon": "📚",
        "color": "#9b59b6",
    },
}


# ─────────────────────────────────────────────
# SUGGESTED GOALS LIBRARY (linked to screening)
# ─────────────────────────────────────────────
SUGGESTED_GOALS = {
    "social_communication": [
        {
            "goal_text": "Respond to name within 3 seconds",
            "domain": "social",
            "target_behavior": "Orients head/eyes toward speaker within 3 seconds of name being called",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "Per screening results",
            "target_value": "8/10 trials",
        },
        {
            "goal_text": "Initiate joint attention by pointing",
            "domain": "communication",
            "target_behavior": "Points to an object/event to share interest (not to request)",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "0/10 trials",
            "target_value": "8/10 trials",
        },
        {
            "goal_text": "Follow a point to a distal object",
            "domain": "social",
            "target_behavior": "Looks in the direction of an adult's point to an object across the room",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "Per screening results",
            "target_value": "8/10 trials",
        },
    ],
    "nonverbal_communication": [
        {
            "goal_text": "Maintain eye contact for 3 seconds during interaction",
            "domain": "social",
            "target_behavior": "Makes and sustains eye contact for at least 3 seconds during social exchange",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "Per screening results",
            "target_value": "8/10 opportunities",
        },
        {
            "goal_text": "Reciprocate a smile",
            "domain": "social",
            "target_behavior": "Smiles in response to an adult's smile within 3 seconds",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "Per screening results",
            "target_value": "8/10 opportunities",
        },
        {
            "goal_text": "Imitate one motor action on request",
            "domain": "communication",
            "target_behavior": "Imitates a gross motor action (clap, wave, stomp) when modeled by adult",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "0/10 trials",
            "target_value": "8/10 trials",
        },
    ],
    "repetitive_behaviors": [
        {
            "goal_text": "Accept a transition with visual support",
            "domain": "behavior",
            "target_behavior": "Transitions between activities within 1 minute when shown a visual schedule/timer",
            "mastery_criteria": "80% across 5 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 5,
            "baseline_value": "Per parent report",
            "target_value": "8/10 transitions",
        },
        {
            "goal_text": "Use a replacement behavior when dysregulated",
            "domain": "behavior",
            "target_behavior": "Uses a taught calming strategy (deep breaths, squeeze ball) instead of repetitive motor behavior when prompted",
            "mastery_criteria": "70% across 5 consecutive sessions",
            "mastery_percentage": 70.0,
            "mastery_consecutive": 5,
            "baseline_value": "0/10 opportunities",
            "target_value": "7/10 opportunities",
        },
    ],
    "sensory_processing": [
        {
            "goal_text": "Tolerate a non-preferred sensory experience for 30 seconds",
            "domain": "adaptive",
            "target_behavior": "Remains calm (no crying/fleeing) during a non-preferred sensory activity for 30+ seconds",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "Per parent report",
            "target_value": "8/10 exposures",
        },
    ],
    "auditory_response": [
        {
            "goal_text": "Respond to name from across the room",
            "domain": "social",
            "target_behavior": "Orients toward speaker within 5 seconds when name is called from 10+ feet away",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "Per screening results",
            "target_value": "8/10 trials",
        },
        {
            "goal_text": "Follow a one-step verbal direction without gestures",
            "domain": "communication",
            "target_behavior": "Completes a one-step direction (e.g., 'Give me the ball') without gestural support",
            "mastery_criteria": "80% across 3 consecutive sessions",
            "mastery_percentage": 80.0,
            "mastery_consecutive": 3,
            "baseline_value": "Per parent report",
            "target_value": "8/10 directions",
        },
    ],
}


# ─────────────────────────────────────────────
# TRACKER CLASS
# ─────────────────────────────────────────────
class TherapyGoalTracker:
    """
    Manages therapy goals for a specific subject.
    Persists data using the data_store module.
    """

    def __init__(self, subject_id: str = "Anonymous"):
        self.subject_id = subject_id
        self.goals: Dict[str, TherapyGoal] = {}
        self._load()

    def _get_filename(self) -> str:
        return get_subject_file(self.subject_id, "therapy_goals")

    def _load(self):
        """Load saved goals from disk."""
        data = load_json(self._get_filename(), default={})
        for gid, gdata in data.items():
            trial_data = []
            for tdata in gdata.get("trial_data", []):
                trial_data.append(TrialEntry(
                    date=tdata.get("date", ""),
                    successful_trials=tdata.get("successful_trials", 0),
                    total_trials=tdata.get("total_trials", 10),
                    prompt_level=tdata.get("prompt_level", "independent"),
                    notes=tdata.get("notes", ""),
                    environment=tdata.get("environment", ""),
                ))

            self.goals[gid] = TherapyGoal(
                goal_id=gid,
                goal_text=gdata.get("goal_text", ""),
                domain=gdata.get("domain", ""),
                target_behavior=gdata.get("target_behavior", ""),
                mastery_criteria=gdata.get("mastery_criteria", ""),
                mastery_percentage=gdata.get("mastery_percentage", 80.0),
                mastery_consecutive=gdata.get("mastery_consecutive", 3),
                phase=gdata.get("phase", "baseline"),
                baseline_value=gdata.get("baseline_value", ""),
                target_value=gdata.get("target_value", ""),
                start_date=gdata.get("start_date", ""),
                trial_data=trial_data,
                status=gdata.get("status", "active"),
                created_at=gdata.get("created_at", ""),
            )

    def save(self):
        """Persist all goals to disk."""
        data = {}
        for gid, goal in self.goals.items():
            trial_list = []
            for t in goal.trial_data:
                trial_list.append({
                    "date": t.date,
                    "successful_trials": t.successful_trials,
                    "total_trials": t.total_trials,
                    "prompt_level": t.prompt_level,
                    "notes": t.notes,
                    "environment": t.environment,
                })

            data[gid] = {
                "goal_text": goal.goal_text,
                "domain": goal.domain,
                "target_behavior": goal.target_behavior,
                "mastery_criteria": goal.mastery_criteria,
                "mastery_percentage": goal.mastery_percentage,
                "mastery_consecutive": goal.mastery_consecutive,
                "phase": goal.phase,
                "baseline_value": goal.baseline_value,
                "target_value": goal.target_value,
                "start_date": goal.start_date,
                "trial_data": trial_list,
                "status": goal.status,
                "created_at": goal.created_at,
            }
        save_json(self._get_filename(), data)

    def add_goal(
        self,
        goal_text: str,
        domain: str,
        target_behavior: str,
        mastery_criteria: str = "80% across 3 consecutive sessions",
        mastery_percentage: float = 80.0,
        mastery_consecutive: int = 3,
        baseline_value: str = "",
        target_value: str = "",
    ) -> str:
        """Add a new goal. Returns goal_id."""
        gid = f"goal_{int(datetime.now().timestamp())}_{len(self.goals)}"

        self.goals[gid] = TherapyGoal(
            goal_id=gid,
            goal_text=goal_text,
            domain=domain,
            target_behavior=target_behavior,
            mastery_criteria=mastery_criteria,
            mastery_percentage=mastery_percentage,
            mastery_consecutive=mastery_consecutive,
            phase="baseline",
            baseline_value=baseline_value,
            target_value=target_value,
            start_date=datetime.now().strftime("%Y-%m-%d"),
        )
        self.save()
        return gid

    def add_goal_from_suggestion(self, suggestion: Dict) -> str:
        """Add a goal from the SUGGESTED_GOALS templates."""
        return self.add_goal(
            goal_text=suggestion["goal_text"],
            domain=suggestion["domain"],
            target_behavior=suggestion["target_behavior"],
            mastery_criteria=suggestion["mastery_criteria"],
            mastery_percentage=suggestion["mastery_percentage"],
            mastery_consecutive=suggestion["mastery_consecutive"],
            baseline_value=suggestion.get("baseline_value", ""),
            target_value=suggestion.get("target_value", ""),
        )

    def log_trial(
        self,
        goal_id: str,
        successful_trials: int,
        total_trials: int = 10,
        prompt_level: str = "independent",
        notes: str = "",
        environment: str = "home",
        date: Optional[str] = None,
    ) -> bool:
        """Log a trial data entry for a goal. Returns True on success."""
        goal = self.goals.get(goal_id)
        if not goal:
            return False

        if prompt_level not in PROMPT_LEVELS:
            prompt_level = "independent"

        entry = TrialEntry(
            date=date or datetime.now().strftime("%Y-%m-%d"),
            successful_trials=min(successful_trials, total_trials),
            total_trials=total_trials,
            prompt_level=prompt_level,
            notes=notes,
            environment=environment,
        )

        goal.trial_data.append(entry)

        # Auto-detect mastery
        if goal.is_mastered and goal.phase == "acquisition":
            goal.phase = "maintenance"
            goal.status = "mastered"

        self.save()
        return True

    def update_phase(self, goal_id: str, new_phase: str) -> bool:
        """Manually update a goal's phase."""
        goal = self.goals.get(goal_id)
        if not goal or new_phase not in PHASE_INFO:
            return False
        goal.phase = new_phase
        self.save()
        return True

    def update_status(self, goal_id: str, new_status: str) -> bool:
        """Update a goal's status."""
        valid = {"active", "mastered", "on_hold", "discontinued"}
        goal = self.goals.get(goal_id)
        if not goal or new_status not in valid:
            return False
        goal.status = new_status
        self.save()
        return True

    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal."""
        if goal_id in self.goals:
            del self.goals[goal_id]
            self.save()
            return True
        return False

    def get_goal(self, goal_id: str) -> Optional[TherapyGoal]:
        """Get a single goal by ID."""
        return self.goals.get(goal_id)

    def get_active_goals(self) -> List[TherapyGoal]:
        """Return all active goals sorted by domain."""
        return sorted(
            [g for g in self.goals.values() if g.status == "active"],
            key=lambda x: x.domain,
        )

    def get_goals_by_domain(self, domain: str) -> List[TherapyGoal]:
        """Return all goals for a specific domain."""
        return [g for g in self.goals.values() if g.domain == domain]

    def get_goals_by_status(self, status: str) -> List[TherapyGoal]:
        """Return all goals with a specific status."""
        return [g for g in self.goals.values() if g.status == status]

    def get_mastered_goals(self) -> List[TherapyGoal]:
        """Return all mastered goals."""
        return [g for g in self.goals.values() if g.status == "mastered"]

    def get_suggested_goals(self, matched_domains: List[str]) -> List[Dict]:
        """
        Return suggested goals based on screening-matched domains.
        
        Parameters:
            matched_domains: List of domain keys from ResourceDirectory.match_from_*
        """
        suggestions = []
        seen_texts = set()

        for domain_key in matched_domains:
            domain_suggestions = SUGGESTED_GOALS.get(domain_key, [])
            for suggestion in domain_suggestions:
                if suggestion["goal_text"] not in seen_texts:
                    seen_texts.add(suggestion["goal_text"])
                    suggestions.append({
                        **suggestion,
                        "source_domain": domain_key,
                    })

        return suggestions

    def generate_progress_summary(self) -> Dict:
        """Generate an overall progress summary across all goals."""
        all_goals = list(self.goals.values())
        if not all_goals:
            return {
                "total_goals": 0,
                "active": 0,
                "mastered": 0,
                "on_hold": 0,
                "discontinued": 0,
                "total_sessions": 0,
                "average_latest_percentage": 0.0,
                "domain_breakdown": {},
                "improving_count": 0,
                "stable_count": 0,
                "declining_count": 0,
            }

        status_counts = {"active": 0, "mastered": 0, "on_hold": 0, "discontinued": 0}
        trend_counts = {"improving": 0, "stable": 0, "declining": 0, "insufficient_data": 0}
        domain_data = {}
        total_sessions = 0
        percentages = []

        for goal in all_goals:
            status_counts[goal.status] = status_counts.get(goal.status, 0) + 1
            trend_counts[goal.trend] = trend_counts.get(goal.trend, 0) + 1
            total_sessions += goal.session_count

            if goal.trial_data:
                percentages.append(goal.latest_percentage)

            if goal.domain not in domain_data:
                domain_data[goal.domain] = {"count": 0, "mastered": 0, "avg_pct": []}
            domain_data[goal.domain]["count"] += 1
            if goal.status == "mastered":
                domain_data[goal.domain]["mastered"] += 1
            if goal.trial_data:
                domain_data[goal.domain]["avg_pct"].append(goal.latest_percentage)

        # Compute domain averages
        for domain_key, ddata in domain_data.items():
            pcts = ddata.pop("avg_pct")
            ddata["average_percentage"] = sum(pcts) / len(pcts) if pcts else 0.0

        avg_pct = sum(percentages) / len(percentages) if percentages else 0.0

        return {
            "total_goals": len(all_goals),
            "active": status_counts.get("active", 0),
            "mastered": status_counts.get("mastered", 0),
            "on_hold": status_counts.get("on_hold", 0),
            "discontinued": status_counts.get("discontinued", 0),
            "total_sessions": total_sessions,
            "average_latest_percentage": avg_pct,
            "domain_breakdown": domain_data,
            "improving_count": trend_counts.get("improving", 0),
            "stable_count": trend_counts.get("stable", 0),
            "declining_count": trend_counts.get("declining", 0),
        }