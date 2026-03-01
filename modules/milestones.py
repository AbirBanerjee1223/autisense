# modules/milestones.py
"""
Developmental Milestone Tracker.
Based on CDC "Learn the Signs. Act Early." program (public domain).
Updated to reflect the 2022 CDC milestone revision.

Source: https://www.cdc.gov/ncbddd/actearly/milestones/
Reference: Zubler et al. (2022). Pediatrics, 149(3), e2021052138.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from modules.data_store import load_json, save_json, get_subject_file


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class Milestone:
    id: str
    text: str
    category: str            # "social_emotional", "language_communication", "cognitive", "movement"
    age_months: int
    icon: str


@dataclass
class MilestoneProgress:
    milestone_id: str
    achieved: bool
    date_recorded: str
    notes: str = ""


@dataclass
class MilestoneReport:
    age_group: str
    total_milestones: int
    achieved: int
    not_achieved: int
    not_assessed: int
    achievement_rate: float
    concern_areas: List[str]
    categories_breakdown: Dict[str, Dict]


# ─────────────────────────────────────────────
# CDC MILESTONE DATABASE (2022 Revision)
# ─────────────────────────────────────────────
MILESTONES_DB: Dict[str, List[Milestone]] = {
    "9_months": [
        Milestone("9m_se_1", "Is shy, clingy, or fearful around strangers", "social_emotional", 9, "😊"),
        Milestone("9m_se_2", "Shows several facial expressions, like happy, sad, angry, and surprised", "social_emotional", 9, "🎭"),
        Milestone("9m_se_3", "Looks when you call her name", "social_emotional", 9, "👂"),
        Milestone("9m_se_4", "Reacts when you leave (looks, reaches for you, or cries)", "social_emotional", 9, "😢"),
        Milestone("9m_se_5", "Smiles or laughs when you play peek-a-boo", "social_emotional", 9, "😄"),
        Milestone("9m_lc_1", "Makes different sounds like 'mamamama' and 'babababa'", "language_communication", 9, "🗣️"),
        Milestone("9m_lc_2", "Lifts arms up to be picked up", "language_communication", 9, "🙌"),
        Milestone("9m_cg_1", "Looks for objects when dropped out of sight (like his spoon or toy)", "cognitive", 9, "🔍"),
        Milestone("9m_cg_2", "Bangs two things together", "cognitive", 9, "🥁"),
        Milestone("9m_mv_1", "Gets to a sitting position by herself", "movement", 9, "🧒"),
        Milestone("9m_mv_2", "Moves things from one hand to her other hand", "movement", 9, "✋"),
        Milestone("9m_mv_3", "Uses fingers to 'rake' food towards himself", "movement", 9, "🍪"),
    ],
    "12_months": [
        Milestone("12m_se_1", "Plays games with you, like pat-a-cake", "social_emotional", 12, "👏"),
        Milestone("12m_se_2", "Waves bye-bye", "social_emotional", 12, "👋"),
        Milestone("12m_se_3", "Calls a parent 'mama' or 'dada' or another special name", "social_emotional", 12, "👨‍👩‍👧"),
        Milestone("12m_se_4", "Understands 'no' (pauses briefly or stops when you say it)", "social_emotional", 12, "🛑"),
        Milestone("12m_lc_1", "Puts something in a container, like a block in a cup", "language_communication", 12, "📦"),
        Milestone("12m_lc_2", "Looks for things he sees you hide, like a toy under a blanket", "language_communication", 12, "🔎"),
        Milestone("12m_cg_1", "Pulls up to stand", "cognitive", 12, "🧍"),
        Milestone("12m_cg_2", "Walks, holding on to furniture", "cognitive", 12, "🚶"),
        Milestone("12m_mv_1", "Drinks from a cup without a lid, as you hold it", "movement", 12, "🥤"),
        Milestone("12m_mv_2", "Picks things up between thumb and pointer finger, like small bits of food", "movement", 12, "🤏"),
    ],
    "18_months": [
        Milestone("18m_se_1", "Moves away from you, but looks to make sure you are close by", "social_emotional", 18, "👀"),
        Milestone("18m_se_2", "Points to show you something interesting", "social_emotional", 18, "☝️"),
        Milestone("18m_se_3", "Puts hands out for you to wash them", "social_emotional", 18, "🧼"),
        Milestone("18m_se_4", "Looks at a few pages in a book with you", "social_emotional", 18, "📖"),
        Milestone("18m_se_5", "Helps you dress him by pushing arm through sleeve or lifting up foot", "social_emotional", 18, "👕"),
        Milestone("18m_lc_1", "Tries to say three or more words besides 'mama' or 'dada'", "language_communication", 18, "🗣️"),
        Milestone("18m_lc_2", "Follows one-step directions without any gestures", "language_communication", 18, "📋"),
        Milestone("18m_cg_1", "Copies you doing chores, like sweeping with a broom", "cognitive", 18, "🧹"),
        Milestone("18m_cg_2", "Plays with toys in a simple way, like pushing a toy car", "cognitive", 18, "🚗"),
        Milestone("18m_mv_1", "Walks without holding on to anyone or anything", "movement", 18, "🚶‍♂️"),
        Milestone("18m_mv_2", "Scribbles", "movement", 18, "✏️"),
        Milestone("18m_mv_3", "Can use a spoon", "movement", 18, "🥄"),
        Milestone("18m_mv_4", "Tries to use switches, knobs, or buttons on a toy", "movement", 18, "🔘"),
        Milestone("18m_mv_5", "Stacks at least two small objects, like blocks", "movement", 18, "🧱"),
    ],
    "24_months": [
        Milestone("24m_se_1", "Notices when others are hurt or upset, like pausing or looking sad when someone is crying", "social_emotional", 24, "🥺"),
        Milestone("24m_se_2", "Looks at your face to see how to react in a new situation", "social_emotional", 24, "👀"),
        Milestone("24m_lc_1", "Points to things in a book when you ask, like 'Where is the bear?'", "language_communication", 24, "📚"),
        Milestone("24m_lc_2", "Says at least two words together, like 'More milk'", "language_communication", 24, "💬"),
        Milestone("24m_lc_3", "Points to at least two body parts when you ask him to show you", "language_communication", 24, "🦶"),
        Milestone("24m_lc_4", "Uses more gestures than just waving and pointing, like blowing a kiss or nodding yes", "language_communication", 24, "😘"),
        Milestone("24m_cg_1", "Holds something in one hand while using the other hand", "cognitive", 24, "🤲"),
        Milestone("24m_cg_2", "Tries to use switches, knobs, or buttons on a toy", "cognitive", 24, "🎛️"),
        Milestone("24m_cg_3", "Plays with more than one toy at the same time, like putting toy food on a toy plate", "cognitive", 24, "🍽️"),
        Milestone("24m_mv_1", "Kicks a ball", "movement", 24, "⚽"),
        Milestone("24m_mv_2", "Runs", "movement", 24, "🏃"),
        Milestone("24m_mv_3", "Walks (not climbs) up a few stairs with or without help", "movement", 24, "🪜"),
        Milestone("24m_mv_4", "Eats with a spoon", "movement", 24, "🥄"),
    ],
    "36_months": [
        Milestone("36m_se_1", "Calms down within 10 minutes after you leave her, like at childcare drop off", "social_emotional", 36, "😌"),
        Milestone("36m_se_2", "Notices other children and joins them to play", "social_emotional", 36, "👫"),
        Milestone("36m_lc_1", "Talks with you in conversation using at least two back-and-forth exchanges", "language_communication", 36, "💬"),
        Milestone("36m_lc_2", "Asks 'who', 'what', 'where', or 'why' questions", "language_communication", 36, "❓"),
        Milestone("36m_lc_3", "Says what action is happening in a picture or book when asked, like 'running', 'eating', or 'playing'", "language_communication", 36, "📖"),
        Milestone("36m_lc_4", "Says first name, when asked", "language_communication", 36, "🏷️"),
        Milestone("36m_lc_5", "Talks well enough for others to understand, most of the time", "language_communication", 36, "🗣️"),
        Milestone("36m_cg_1", "Draws a circle, when you show him how", "cognitive", 36, "⭕"),
        Milestone("36m_cg_2", "Avoids touching hot objects, like a stove, when you warn her", "cognitive", 36, "🔥"),
        Milestone("36m_mv_1", "Strings items together, like large beads or macaroni", "movement", 36, "📿"),
        Milestone("36m_mv_2", "Puts on some clothes by himself, like loose pants or a jacket", "movement", 36, "🧥"),
        Milestone("36m_mv_3", "Uses a fork", "movement", 36, "🍴"),
    ],
    "48_months": [
        Milestone("48m_se_1", "Pretends to be something else during play (teacher, superhero, dog)", "social_emotional", 48, "🦸"),
        Milestone("48m_se_2", "Asks to go play with children if none are around", "social_emotional", 48, "🙋"),
        Milestone("48m_se_3", "Comforts others who are hurt or sad, like hugging a crying friend", "social_emotional", 48, "🤗"),
        Milestone("48m_se_4", "Avoids danger, like not jumping from tall heights at the playground", "social_emotional", 48, "⚠️"),
        Milestone("48m_se_5", "Likes to be a 'helper'", "social_emotional", 48, "🙌"),
        Milestone("48m_se_6", "Changes behavior based on where she is (place of worship, library, playground)", "social_emotional", 48, "🔄"),
        Milestone("48m_lc_1", "Says sentences with four or more words", "language_communication", 48, "📝"),
        Milestone("48m_lc_2", "Says some words from a song, story, or nursery rhyme", "language_communication", 48, "🎵"),
        Milestone("48m_lc_3", "Talks about at least one thing that happened during the day", "language_communication", 48, "📅"),
        Milestone("48m_lc_4", "Answers simple questions like 'What is a coat for?' or 'What is a crayon for?'", "language_communication", 48, "💡"),
        Milestone("48m_cg_1", "Names a few colors of items", "cognitive", 48, "🎨"),
        Milestone("48m_cg_2", "Tells what comes next in a well-known story", "cognitive", 48, "📖"),
        Milestone("48m_cg_3", "Draws a person with three or more body parts", "cognitive", 48, "🧑‍🎨"),
        Milestone("48m_mv_1", "Catches a large ball most of the time", "movement", 48, "🏐"),
        Milestone("48m_mv_2", "Serves himself food or pours water, with adult supervision", "movement", 48, "🍽️"),
        Milestone("48m_mv_3", "Unbuttons some buttons", "movement", 48, "👔"),
        Milestone("48m_mv_4", "Holds crayon or pencil between fingers and thumb (not a fist)", "movement", 48, "✏️"),
    ],
}

# Category display info
CATEGORY_INFO = {
    "social_emotional": {"label": "Social & Emotional", "icon": "💛", "color": "#fdcb6e"},
    "language_communication": {"label": "Language & Communication", "icon": "💬", "color": "#74b9ff"},
    "cognitive": {"label": "Cognitive", "icon": "🧠", "color": "#a29bfe"},
    "movement": {"label": "Movement & Physical", "icon": "🏃", "color": "#55efc4"},
}

AGE_GROUP_LABELS = {
    "9_months": "9 Months",
    "12_months": "12 Months",
    "18_months": "18 Months",
    "24_months": "24 Months",
    "36_months": "3 Years",
    "48_months": "4 Years",
}


# ─────────────────────────────────────────────
# TRACKER CLASS
# ─────────────────────────────────────────────
class MilestoneTracker:
    """
    Tracks developmental milestones for a specific subject.
    Persists data using the data_store module.
    """

    def __init__(self, subject_id: str = "Anonymous"):
        self.subject_id = subject_id
        self.progress: Dict[str, MilestoneProgress] = {}
        self._load()

    def _get_filename(self) -> str:
        return get_subject_file(self.subject_id, "milestones")

    def _load(self):
        """Load saved progress from disk."""
        data = load_json(self._get_filename(), default={})
        for mid, entry in data.items():
            self.progress[mid] = MilestoneProgress(
                milestone_id=mid,
                achieved=entry.get("achieved", False),
                date_recorded=entry.get("date_recorded", ""),
                notes=entry.get("notes", ""),
            )

    def save(self):
        """Persist current progress to disk."""
        data = {}
        for mid, prog in self.progress.items():
            data[mid] = {
                "achieved": prog.achieved,
                "date_recorded": prog.date_recorded,
                "notes": prog.notes,
            }
        save_json(self._get_filename(), data)

    def set_milestone(self, milestone_id: str, achieved: bool, notes: str = ""):
        """Record whether a milestone has been achieved."""
        self.progress[milestone_id] = MilestoneProgress(
            milestone_id=milestone_id,
            achieved=achieved,
            date_recorded=datetime.now().isoformat(),
            notes=notes,
        )
        self.save()

    def get_status(self, milestone_id: str) -> Optional[bool]:
        """Get achievement status. Returns None if not yet assessed."""
        if milestone_id in self.progress:
            return self.progress[milestone_id].achieved
        return None

    def get_milestones_for_age(self, age_group: str) -> List[Milestone]:
        """Get all milestones for a specific age group."""
        return MILESTONES_DB.get(age_group, [])

    def generate_report(self, age_group: str) -> MilestoneReport:
        """Generate a summary report for a specific age group."""
        milestones = self.get_milestones_for_age(age_group)
        if not milestones:
            return MilestoneReport(
                age_group=age_group,
                total_milestones=0,
                achieved=0,
                not_achieved=0,
                not_assessed=0,
                achievement_rate=0.0,
                concern_areas=[],
                categories_breakdown={},
            )

        achieved = 0
        not_achieved = 0
        not_assessed = 0
        category_data = {}

        for m in milestones:
            status = self.get_status(m.id)

            if m.category not in category_data:
                category_data[m.category] = {"total": 0, "achieved": 0, "not_achieved": 0, "not_assessed": 0}

            category_data[m.category]["total"] += 1

            if status is True:
                achieved += 1
                category_data[m.category]["achieved"] += 1
            elif status is False:
                not_achieved += 1
                category_data[m.category]["not_achieved"] += 1
            else:
                not_assessed += 1
                category_data[m.category]["not_assessed"] += 1

        total = len(milestones)
        assessed = achieved + not_achieved
        rate = (achieved / assessed * 100) if assessed > 0 else 0.0

        # Identify concern areas
        concern_areas = []
        for cat, data in category_data.items():
            cat_assessed = data["achieved"] + data["not_achieved"]
            if cat_assessed > 0:
                cat_rate = data["achieved"] / cat_assessed
                if cat_rate < 0.5:
                    cat_info = CATEGORY_INFO.get(cat, {"label": cat})
                    concern_areas.append(cat_info["label"])

        return MilestoneReport(
            age_group=age_group,
            total_milestones=total,
            achieved=achieved,
            not_achieved=not_achieved,
            not_assessed=not_assessed,
            achievement_rate=rate,
            concern_areas=concern_areas,
            categories_breakdown=category_data,
        )

    def get_all_concern_areas(self) -> List[str]:
        """Check all age groups and return categories with low achievement."""
        all_concerns = set()
        for age_group in MILESTONES_DB:
            report = self.generate_report(age_group)
            all_concerns.update(report.concern_areas)
        return sorted(all_concerns)

    @staticmethod
    def get_age_groups() -> Dict[str, str]:
        """Return available age groups with labels."""
        return dict(AGE_GROUP_LABELS)

    @staticmethod
    def get_category_info() -> Dict:
        """Return category display information."""
        return dict(CATEGORY_INFO)