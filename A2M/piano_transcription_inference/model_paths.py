import sys
from pathlib import Path

MODEL_URL = "https://downloads.justagwas.com/a2m/PianoModel.pth"
MODEL_DOWNLOAD_URL = MODEL_URL
MODEL_FILENAME = "PianoModel.pth"
MODEL_MIN_BYTES = 160_000_000
MODEL_DIR = Path.home() / "piano_transcription_inference_data"


def get_app_dir(app_file=None):
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    if app_file:
        return Path(app_file).resolve().parent
    return Path(__file__).resolve().parent.parent


def find_existing_model(base_dir):
    base = Path(base_dir)
    candidate = base / MODEL_FILENAME
    if candidate.exists() and candidate.stat().st_size >= MODEL_MIN_BYTES:
        return candidate
    return None


def get_model_candidate_dirs(app_file=None):
    app_dir = get_app_dir(app_file=app_file)
    dirs = [app_dir]
    if app_dir.resolve() != MODEL_DIR.resolve():
        dirs.append(MODEL_DIR)
    return dirs


def get_existing_model_path(app_file=None):
    for base_dir in get_model_candidate_dirs(app_file=app_file):
        existing = find_existing_model(base_dir)
        if existing:
            return existing
    return None


def resolve_checkpoint_path(checkpoint_path=None, app_file=None):
    if checkpoint_path:
        return str(checkpoint_path)
    existing = get_existing_model_path(app_file=app_file)
    if existing:
        return str(existing)
    return str(get_app_dir(app_file=app_file) / MODEL_FILENAME)
