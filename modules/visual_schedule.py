# modules/visual_schedule.py
"""
Visual Schedule Builder.
Creates printable visual schedules using emoji icons.
Based on TEACCH structured teaching principles.

Reference: Mesibov, G., Shea, V., & Schopler, E. (2005).
The TEACCH Approach to Autism Spectrum Disorders. Springer.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from modules.data_store import load_json, save_json, get_subject_file


SCHEDULE_DIR = Path("reports")
SCHEDULE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ACTIVITY DATABASE
# ─────────────────────────────────────────────
@dataclass
class Activity:
    id: str
    name: str
    icon: str
    category: str           # "morning", "meals", "learning", "therapy", "play", "evening", "transitions"
    color: str              # hex color for the card background
    default_duration: int   # minutes


ACTIVITY_LIBRARY: Dict[str, List[Activity]] = {
    "Morning Routine": [
        Activity("wake_up", "Wake Up", "🌅", "morning", "#FFF9C4", 5),
        Activity("get_dressed", "Get Dressed", "👕", "morning", "#FFF9C4", 10),
        Activity("brush_teeth_am", "Brush Teeth", "🪥", "morning", "#FFF9C4", 5),
        Activity("wash_face", "Wash Face", "🧼", "morning", "#FFF9C4", 5),
        Activity("comb_hair", "Comb Hair", "💇", "morning", "#FFF9C4", 5),
    ],
    "Meals & Snacks": [
        Activity("breakfast", "Breakfast", "🥣", "meals", "#DCEDC8", 20),
        Activity("morning_snack", "Morning Snack", "🍎", "meals", "#DCEDC8", 10),
        Activity("lunch", "Lunch", "🥪", "meals", "#DCEDC8", 25),
        Activity("afternoon_snack", "Afternoon Snack", "🧃", "meals", "#DCEDC8", 10),
        Activity("dinner", "Dinner", "🍽️", "meals", "#DCEDC8", 25),
        Activity("drink_water", "Drink Water", "💧", "meals", "#DCEDC8", 2),
    ],
    "Learning & School": [
        Activity("school", "School", "🏫", "learning", "#BBDEFB", 180),
        Activity("reading", "Reading Time", "📖", "learning", "#BBDEFB", 15),
        Activity("coloring", "Coloring / Drawing", "🎨", "learning", "#BBDEFB", 20),
        Activity("numbers", "Number Practice", "🔢", "learning", "#BBDEFB", 15),
        Activity("letters", "Letter Practice", "🔤", "learning", "#BBDEFB", 15),
        Activity("homework", "Homework", "📝", "learning", "#BBDEFB", 20),
        Activity("music", "Music Time", "🎵", "learning", "#BBDEFB", 15),
    ],
    "Therapy & Appointments": [
        Activity("speech_therapy", "Speech Therapy", "🗣️", "therapy", "#E1BEE7", 45),
        Activity("ot_therapy", "Occupational Therapy", "🧩", "therapy", "#E1BEE7", 45),
        Activity("aba_therapy", "ABA Therapy", "📋", "therapy", "#E1BEE7", 60),
        Activity("play_therapy", "Play Therapy", "🧸", "therapy", "#E1BEE7", 45),
        Activity("doctor_visit", "Doctor Visit", "🏥", "therapy", "#E1BEE7", 60),
        Activity("group_therapy", "Group Session", "👫", "therapy", "#E1BEE7", 45),
    ],
    "Play & Free Time": [
        Activity("free_play", "Free Play", "🎮", "play", "#B2EBF2", 30),
        Activity("outdoor_play", "Outdoor Play", "🌳", "play", "#B2EBF2", 30),
        Activity("sensory_play", "Sensory Play", "🫧", "play", "#B2EBF2", 20),
        Activity("building_blocks", "Building Blocks", "🧱", "play", "#B2EBF2", 20),
        Activity("puzzle", "Puzzle Time", "🧩", "play", "#B2EBF2", 15),
        Activity("playground", "Playground", "🛝", "play", "#B2EBF2", 30),
        Activity("screen_time", "Screen Time", "📺", "play", "#B2EBF2", 20),
        Activity("pet_time", "Time with Pets", "🐾", "play", "#B2EBF2", 15),
    ],
    "Evening Routine": [
        Activity("bath", "Bath Time", "🛁", "evening", "#F8BBD0", 20),
        Activity("brush_teeth_pm", "Brush Teeth", "🪥", "evening", "#F8BBD0", 5),
        Activity("pajamas", "Put on Pajamas", "👔", "evening", "#F8BBD0", 5),
        Activity("story_time", "Story Time", "📚", "evening", "#F8BBD0", 15),
        Activity("goodnight", "Say Goodnight", "🌙", "evening", "#F8BBD0", 5),
        Activity("bedtime", "Bedtime", "😴", "evening", "#F8BBD0", 0),
    ],
    "Transitions & Helpers": [
        Activity("timer_warning", "Timer Warning", "⏰", "transitions", "#FFE0B2", 2),
        Activity("clean_up", "Clean Up", "🧹", "transitions", "#FFE0B2", 10),
        Activity("car_ride", "Car Ride", "🚗", "transitions", "#FFE0B2", 15),
        Activity("waiting", "Waiting Time", "⏳", "transitions", "#FFE0B2", 5),
        Activity("deep_breath", "Deep Breaths", "🌬️", "transitions", "#FFE0B2", 3),
        Activity("quiet_time", "Quiet Time", "🤫", "transitions", "#FFE0B2", 10),
        Activity("movement_break", "Movement Break", "🤸", "transitions", "#FFE0B2", 5),
        Activity("check_schedule", "Check Schedule", "📋", "transitions", "#FFE0B2", 1),
    ],
}


# ─────────────────────────────────────────────
# SCHEDULE DATA CLASS
# ─────────────────────────────────────────────
@dataclass
class ScheduleItem:
    """A single item in the schedule with optional time and notes."""
    activity_id: str
    activity_name: str
    icon: str
    color: str
    position: int
    time_label: str = ""       # e.g., "8:00 AM"
    duration_minutes: int = 0
    notes: str = ""
    completed: bool = False


@dataclass
class DailySchedule:
    """A complete daily schedule."""
    schedule_id: str
    name: str                   # e.g., "Monday Schedule", "Therapy Day"
    items: List[ScheduleItem] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def completed_items(self) -> int:
        return sum(1 for item in self.items if item.completed)

    @property
    def completion_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100


# ─────────────────────────────────────────────
# TEMPLATE SCHEDULES
# ─────────────────────────────────────────────
SCHEDULE_TEMPLATES = {
    "typical_weekday": {
        "name": "Typical Weekday",
        "description": "A standard day with school and one therapy session",
        "items": [
            ("wake_up", "7:00 AM"),
            ("brush_teeth_am", "7:10 AM"),
            ("get_dressed", "7:15 AM"),
            ("breakfast", "7:30 AM"),
            ("car_ride", "8:00 AM"),
            ("school", "8:30 AM"),
            ("lunch", "12:00 PM"),
            ("school", "12:30 PM"),
            ("car_ride", "3:00 PM"),
            ("afternoon_snack", "3:30 PM"),
            ("speech_therapy", "4:00 PM"),
            ("free_play", "5:00 PM"),
            ("dinner", "6:00 PM"),
            ("bath", "7:00 PM"),
            ("pajamas", "7:20 PM"),
            ("story_time", "7:30 PM"),
            ("bedtime", "8:00 PM"),
        ],
    },
    "therapy_day": {
        "name": "Therapy Day",
        "description": "A day with multiple therapy appointments",
        "items": [
            ("wake_up", "7:00 AM"),
            ("brush_teeth_am", "7:10 AM"),
            ("get_dressed", "7:15 AM"),
            ("breakfast", "7:30 AM"),
            ("check_schedule", "8:00 AM"),
            ("speech_therapy", "8:30 AM"),
            ("movement_break", "9:15 AM"),
            ("morning_snack", "9:30 AM"),
            ("ot_therapy", "10:00 AM"),
            ("deep_breath", "10:45 AM"),
            ("free_play", "11:00 AM"),
            ("lunch", "12:00 PM"),
            ("quiet_time", "12:30 PM"),
            ("aba_therapy", "1:00 PM"),
            ("afternoon_snack", "2:00 PM"),
            ("outdoor_play", "2:30 PM"),
            ("dinner", "5:30 PM"),
            ("bath", "6:30 PM"),
            ("story_time", "7:00 PM"),
            ("bedtime", "7:30 PM"),
        ],
    },
    "weekend": {
        "name": "Weekend Day",
        "description": "A relaxed day with more free time",
        "items": [
            ("wake_up", "8:00 AM"),
            ("get_dressed", "8:15 AM"),
            ("breakfast", "8:30 AM"),
            ("outdoor_play", "9:30 AM"),
            ("morning_snack", "10:30 AM"),
            ("sensory_play", "11:00 AM"),
            ("lunch", "12:00 PM"),
            ("quiet_time", "12:30 PM"),
            ("reading", "1:00 PM"),
            ("free_play", "1:30 PM"),
            ("afternoon_snack", "3:00 PM"),
            ("playground", "3:30 PM"),
            ("dinner", "5:30 PM"),
            ("bath", "6:30 PM"),
            ("story_time", "7:15 PM"),
            ("bedtime", "7:30 PM"),
        ],
    },
}


# ─────────────────────────────────────────────
# HELPER: Find activity by ID
# ─────────────────────────────────────────────
def _find_activity(activity_id: str) -> Optional[Activity]:
    """Look up an Activity by its ID across all categories."""
    for category_list in ACTIVITY_LIBRARY.values():
        for act in category_list:
            if act.id == activity_id:
                return act
    return None


def get_all_activities_flat() -> List[Activity]:
    """Return a flat list of all activities."""
    flat = []
    for category_list in ACTIVITY_LIBRARY.values():
        flat.extend(category_list)
    return flat


# ─────────────────────────────────────────────
# SCHEDULE BUILDER CLASS
# ─────────────────────────────────────────────
class VisualScheduleBuilder:
    """
    Builds, manages, and exports visual schedules.
    Persists schedules per subject using data_store.
    """

    def __init__(self, subject_id: str = "Anonymous"):
        self.subject_id = subject_id
        self.schedules: Dict[str, DailySchedule] = {}
        self._load()

    def _get_filename(self) -> str:
        return get_subject_file(self.subject_id, "schedules")

    def _load(self):
        """Load saved schedules from disk."""
        data = load_json(self._get_filename(), default={})
        for sid, sdata in data.items():
            items = []
            for i, item_data in enumerate(sdata.get("items", [])):
                items.append(ScheduleItem(
                    activity_id=item_data.get("activity_id", ""),
                    activity_name=item_data.get("activity_name", ""),
                    icon=item_data.get("icon", "❓"),
                    color=item_data.get("color", "#FFFFFF"),
                    position=i,
                    time_label=item_data.get("time_label", ""),
                    duration_minutes=item_data.get("duration_minutes", 0),
                    notes=item_data.get("notes", ""),
                    completed=item_data.get("completed", False),
                ))
            self.schedules[sid] = DailySchedule(
                schedule_id=sid,
                name=sdata.get("name", "Untitled"),
                items=items,
                created_at=sdata.get("created_at", ""),
                modified_at=sdata.get("modified_at", ""),
            )

    def save(self):
        """Persist all schedules to disk."""
        data = {}
        for sid, schedule in self.schedules.items():
            items_data = []
            for item in schedule.items:
                items_data.append({
                    "activity_id": item.activity_id,
                    "activity_name": item.activity_name,
                    "icon": item.icon,
                    "color": item.color,
                    "time_label": item.time_label,
                    "duration_minutes": item.duration_minutes,
                    "notes": item.notes,
                    "completed": item.completed,
                })
            data[sid] = {
                "name": schedule.name,
                "items": items_data,
                "created_at": schedule.created_at,
                "modified_at": datetime.now().isoformat(),
            }
        save_json(self._get_filename(), data)

    def create_schedule(self, name: str) -> str:
        """Create a new empty schedule. Returns the schedule_id."""
        sid = f"sched_{int(datetime.now().timestamp())}"
        self.schedules[sid] = DailySchedule(
            schedule_id=sid,
            name=name,
            items=[],
        )
        self.save()
        return sid

    def create_from_template(self, template_key: str, custom_name: str = "") -> Optional[str]:
        """
        Create a schedule from a pre-built template.
        Returns schedule_id or None if template not found.
        """
        template = SCHEDULE_TEMPLATES.get(template_key)
        if not template:
            return None

        name = custom_name or template["name"]
        sid = self.create_schedule(name)

        for i, (activity_id, time_label) in enumerate(template["items"]):
            act = _find_activity(activity_id)
            if act:
                self.add_item(
                    schedule_id=sid,
                    activity_id=act.id,
                    time_label=time_label,
                )

        return sid

    def add_item(
        self,
        schedule_id: str,
        activity_id: str,
        time_label: str = "",
        notes: str = "",
        position: Optional[int] = None,
    ) -> bool:
        """Add an activity to a schedule. Returns True on success."""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return False

        act = _find_activity(activity_id)
        if not act:
            return False

        pos = position if position is not None else len(schedule.items)

        item = ScheduleItem(
            activity_id=act.id,
            activity_name=act.name,
            icon=act.icon,
            color=act.color,
            position=pos,
            time_label=time_label,
            duration_minutes=act.default_duration,
            notes=notes,
            completed=False,
        )

        if position is not None and 0 <= position <= len(schedule.items):
            schedule.items.insert(position, item)
            # Reindex positions
            for idx, it in enumerate(schedule.items):
                it.position = idx
        else:
            schedule.items.append(item)
            item.position = len(schedule.items) - 1

        self.save()
        return True

    def remove_item(self, schedule_id: str, position: int) -> bool:
        """Remove an item by position index."""
        schedule = self.schedules.get(schedule_id)
        if not schedule or position < 0 or position >= len(schedule.items):
            return False

        schedule.items.pop(position)
        for idx, it in enumerate(schedule.items):
            it.position = idx

        self.save()
        return True

    def move_item(self, schedule_id: str, from_pos: int, to_pos: int) -> bool:
        """Move an item from one position to another."""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return False
        if from_pos < 0 or from_pos >= len(schedule.items):
            return False
        if to_pos < 0 or to_pos >= len(schedule.items):
            return False

        item = schedule.items.pop(from_pos)
        schedule.items.insert(to_pos, item)
        for idx, it in enumerate(schedule.items):
            it.position = idx

        self.save()
        return True

    def toggle_complete(self, schedule_id: str, position: int) -> bool:
        """Toggle the completed status of an item."""
        schedule = self.schedules.get(schedule_id)
        if not schedule or position < 0 or position >= len(schedule.items):
            return False

        schedule.items[position].completed = not schedule.items[position].completed
        self.save()
        return True

    def set_item_time(self, schedule_id: str, position: int, time_label: str) -> bool:
        """Update the time label for an item."""
        schedule = self.schedules.get(schedule_id)
        if not schedule or position < 0 or position >= len(schedule.items):
            return False

        schedule.items[position].time_label = time_label
        self.save()
        return True

    def set_item_notes(self, schedule_id: str, position: int, notes: str) -> bool:
        """Update notes for an item."""
        schedule = self.schedules.get(schedule_id)
        if not schedule or position < 0 or position >= len(schedule.items):
            return False

        schedule.items[position].notes = notes
        self.save()
        return True

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete an entire schedule."""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self.save()
            return True
        return False

    def get_schedule(self, schedule_id: str) -> Optional[DailySchedule]:
        """Get a schedule by ID."""
        return self.schedules.get(schedule_id)

    def list_schedules(self) -> List[Dict]:
        """Return a list summary of all schedules."""
        result = []
        for sid, schedule in self.schedules.items():
            result.append({
                "schedule_id": sid,
                "name": schedule.name,
                "item_count": schedule.total_items,
                "completed": schedule.completed_items,
                "completion_rate": schedule.completion_rate,
                "created_at": schedule.created_at,
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    def generate_schedule_html(self, schedule_id: str) -> str:
        """
        Generate an HTML representation of the schedule 
        suitable for rendering in Streamlit.
        """
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return "<p>Schedule not found.</p>"

        html_parts = [
            f'<div style="font-family: Nunito, sans-serif; max-width: 600px; margin: auto;">',
            f'<h2 style="text-align:center; color:#2c3e50; margin-bottom:20px;">'
            f'📋 {schedule.name}</h2>',
        ]

        for item in schedule.items:
            done_style = "opacity: 0.5; text-decoration: line-through;" if item.completed else ""
            check = "✅" if item.completed else "⬜"
            time_str = f'<span style="color:#7f8c8d; font-size:0.85rem; margin-right:10px;">{item.time_label}</span>' if item.time_label else ""
            notes_str = f'<div style="font-size:0.75rem; color:#95a5a6; margin-top:3px;">{item.notes}</div>' if item.notes else ""

            html_parts.append(f'''
            <div style="display:flex; align-items:center; gap:12px;
                        background:{item.color}; padding:14px 20px; border-radius:16px;
                        margin:8px 0; box-shadow:0 2px 8px rgba(0,0,0,0.04);
                        font-size:1.1rem; {done_style}">
                <span style="font-size:1.8rem; flex-shrink:0;">{item.icon}</span>
                <div style="flex:1;">
                    {time_str}
                    <b>{item.activity_name}</b>
                    {notes_str}
                </div>
                <span style="font-size:1.2rem;">{check}</span>
            </div>
            ''')

        html_parts.append('</div>')
        return "\n".join(html_parts)

    def export_schedule_pdf(self, schedule_id: str) -> Optional[str]:
        """
        Export a schedule as a printable PDF.
        Returns file path or None.
        """
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return None

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 15, f"My Schedule: {schedule.name}", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8, f"Created: {schedule.created_at[:10]}", ln=True, align="C")
        pdf.ln(10)

        pdf.set_text_color(0, 0, 0)

        # Table header
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(230, 240, 255)
        pdf.cell(10, 10, "#", border=1, fill=True, align="C")
        pdf.cell(30, 10, "Time", border=1, fill=True, align="C")
        pdf.cell(80, 10, "Activity", border=1, fill=True)
        pdf.cell(20, 10, "Min", border=1, fill=True, align="C")
        pdf.cell(50, 10, "Notes", border=1, fill=True)
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 10)
        for i, item in enumerate(schedule.items):
            bg = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*bg)

            status = "[X]" if item.completed else "[ ]"
            pdf.cell(10, 8, f"{i+1}", border=1, fill=True, align="C")
            pdf.cell(30, 8, item.time_label or "-", border=1, fill=True, align="C")
            pdf.cell(80, 8, f"{status} {item.activity_name}", border=1, fill=True)
            dur_str = str(item.duration_minutes) if item.duration_minutes > 0 else "-"
            pdf.cell(20, 8, dur_str, border=1, fill=True, align="C")
            pdf.cell(50, 8, (item.notes or "-")[:25], border=1, fill=True)
            pdf.ln()

        # Footer
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, "Generated by Autisense Visual Schedule Builder", ln=True, align="C")
        pdf.cell(
            0, 6,
            "Based on TEACCH Structured Teaching principles (Mesibov, Shea, & Schopler, 2005)",
            ln=True, align="C"
        )

        filename = f"schedule_{schedule.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = str(SCHEDULE_DIR / filename)
        pdf.output(filepath)
        return filepath

    @staticmethod
    def get_activity_library() -> Dict[str, List[Activity]]:
        """Return the full activity library for UI rendering."""
        return ACTIVITY_LIBRARY

    @staticmethod
    def get_templates() -> Dict[str, Dict]:
        """Return available schedule templates."""
        return SCHEDULE_TEMPLATES