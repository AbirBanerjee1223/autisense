# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).parent
EVIDENCE_DIR = BASE_DIR / "evidence"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
STIMULI_DIR = BASE_DIR / "stimuli"

for d in [EVIDENCE_DIR, REPORTS_DIR, ASSETS_DIR, MODELS_DIR, STIMULI_DIR]:
    d.mkdir(exist_ok=True)

# --- Model Paths ---
FACE_LANDMARKER_MODEL = str(MODELS_DIR / "face_landmarker.task")
POSE_LANDMARKER_MODEL = str(MODELS_DIR / "pose_landmarker_heavy.task")

# --- Stimuli Paths ---
SOCIAL_GEOMETRIC_VIDEO = str(STIMULI_DIR / "social_geometric.mp4")
SMILE_PROMPT_VIDEO = str(STIMULI_DIR / "smile_prompt.mp4")
NAME_CALL_AUDIO = str(STIMULI_DIR / "name_call.wav")

# --- Camera ---
LAPTOP_CAM_INDEX = 0

# --- Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not found. Chatbot uses fallback.")
else:
    print(f"✅ Gemini key loaded (...{GEMINI_API_KEY[-4:]})")

# --- Clinical Baselines (From Published Literature) ---
# These are neurotypical baseline values used to compute
# standard deviations. Sources cited in report.

BASELINES = {
    # Gaze: Neurotypical children look at social stimuli ~70% of time
    # Source: Jones & Klin, Nature 2013
    "social_preference_pct": {"mean": 70.0, "std": 12.0},

    # Name-call response latency in milliseconds
    # Source: Nadig et al. 2007, Campbell et al. 2019
    "name_call_latency_ms": {"mean": 800.0, "std": 300.0},

    # Emotional reciprocity (smile-back rate when prompted)
    # Source: Trevisan et al. 2018 meta-analysis
    "smile_reciprocity_pct": {"mean": 65.0, "std": 15.0},

    # Blink rate (blinks per minute)
    # Source: Sears et al. 1994
    "blinks_per_minute": {"mean": 17.0, "std": 6.0},

    # Gaze away percentage during social interaction
    # Source: Chawarska et al. 2012
    "gaze_away_pct": {"mean": 25.0, "std": 10.0},

    # Expression variance (composite intensity std dev)
    # Source: Owada et al. 2018
    "expression_variance": {"mean": 0.045, "std": 0.015},
}

# --- Session ---
SESSION_DURATION_SECONDS = 90  # Shortened: tighter clinical protocol
FPS_TARGET = 15

# --- Thresholds ---
GAZE_AVOIDANCE_THRESHOLD = 0.35
GAZE_AWAY_DURATION_FLAG = 3.0
BLINK_RATE_LOW = 8
BLINK_RATE_HIGH = 30
REPETITIVE_MOTION_THRESHOLD = 0.6
EMOTION_FLAT_THRESHOLD = 0.7