# model_downloader.py (NEW FILE - put in project root)

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "face_landmarker": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "filename": "face_landmarker.task"
    },
    "pose_landmarker": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "filename": "pose_landmarker_heavy.task"
    }
}


def download_model(model_name: str) -> str:
    """Download a model if not already present. Returns file path."""
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODELS.keys())}"
        )

    model_info = MODELS[model_name]
    filepath = MODELS_DIR / model_info["filename"]

    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✅ {model_name} already downloaded ({size_mb:.1f} MB)")
        return str(filepath)

    print(f"⬇️  Downloading {model_name}...")
    print(f"   URL: {model_info['url']}")
    print(f"   Saving to: {filepath}")

    try:
        urllib.request.urlretrieve(
            model_info["url"],
            str(filepath),
            reporthook=_download_progress
        )
        print(f"\n✅ {model_name} downloaded successfully!")
        return str(filepath)

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if filepath.exists():
            filepath.unlink()
        raise


def _download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(
            f"\r   Progress: {percent:.1f}% "
            f"({mb_down:.1f}/{mb_total:.1f} MB)",
            end="", flush=True
        )


def download_all_models():
    """Download all required models."""
    print("=" * 50)
    print("  MediaPipe Model Downloader")
    print("=" * 50)
    
    paths = {}
    for name in MODELS:
        paths[name] = download_model(name)

    print("\n✅ All models ready!")
    return paths


if __name__ == "__main__":
    download_all_models()