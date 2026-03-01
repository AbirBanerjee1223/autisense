"""
Simple JSON-based persistence layer for care hub tools.
Auto-creates /data directory on first use.
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save_json(filename: str, data) -> None:
    """Save data to a JSON file in the data directory."""
    filepath = DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_json(filename: str, default=None):
    """Load data from a JSON file. Returns default if file doesn't exist."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default if default is not None else {}
    return default if default is not None else {}


def append_entry(filename: str, entry: dict) -> None:
    """Append an entry to a JSON list file."""
    data = load_json(filename, default=[])
    entry['_timestamp'] = datetime.now().isoformat()
    data.append(entry)
    save_json(filename, data)


def get_subject_file(subject_id: str, tool_name: str) -> str:
    """Generate a subject-specific filename for data isolation."""
    safe_id = "".join(c if c.isalnum() else "_" for c in subject_id)
    return f"{safe_id}_{tool_name}.json"