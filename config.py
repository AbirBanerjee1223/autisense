# config.py (UPDATED)

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

EVIDENCE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# --- Model Paths ---
FACE_LANDMARKER_MODEL = str(MODELS_DIR / "face_landmarker.task")
POSE_LANDMARKER_MODEL = str(MODELS_DIR / "pose_landmarker_heavy.task")

# --- Camera Settings ---
LAPTOP_CAM_INDEX = 0
PHONE_CAM_URL = "http://192.168.1.100:8080/video"

# --- Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in .env file!")
    print("   Chatbot will use fallback responses.")
else:
    print(
        f"✅ Gemini API Key loaded "
        f"(ends with ...{GEMINI_API_KEY[-4:]})"
    )

# --- Check Models ---
if not Path(FACE_LANDMARKER_MODEL).exists():
    print("⚠️  Face model not found! Run: python model_downloader.py")
if not Path(POSE_LANDMARKER_MODEL).exists():
    print("⚠️  Pose model not found! Run: python model_downloader.py")

# --- Thresholds ---
GAZE_AVOIDANCE_THRESHOLD = 0.35
GAZE_AWAY_DURATION_FLAG = 3.0
BLINK_RATE_LOW = 8
BLINK_RATE_HIGH = 30
REPETITIVE_MOTION_THRESHOLD = 0.6
STILLNESS_THRESHOLD = 0.02
EMOTION_FLAT_THRESHOLD = 0.7

# --- Session ---
SESSION_DURATION_SECONDS = 120
FPS_TARGET = 15