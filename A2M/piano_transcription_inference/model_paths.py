from __future__ import annotations

import sys
from pathlib import Path

try:
    from a2m.core import model_service as _core_model_service
    from a2m.core.constants import MODEL_FILENAME, MODEL_MIN_BYTES, MODEL_URL
except Exception:
    MODEL_URL = 'https://downloads.justagwas.com/a2m/PianoModel.onnx'
    MODEL_FILENAME = 'PianoModel.onnx'
    MODEL_MIN_BYTES = 20000000
    _core_model_service = None

MODEL_DOWNLOAD_URL = MODEL_URL
MODEL_DIR = _core_model_service.MODEL_DIR if _core_model_service is not None else (Path.home() / 'AppData' / 'Local' / 'A2M' / 'models')


def get_app_dir(app_file=None):
    if getattr(sys, 'frozen', False):
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
    if _core_model_service is not None:
        return [Path(path) for path in _core_model_service.get_model_candidate_dirs()]
    app_dir = get_app_dir(app_file=app_file)
    dirs = [app_dir]
    if app_dir.resolve() != MODEL_DIR.resolve():
        dirs.append(MODEL_DIR)
    return dirs


def get_existing_model_path(app_file=None):
    if _core_model_service is not None:
        existing = _core_model_service.get_existing_model_path()
        if existing:
            return Path(existing)
    for base_dir in get_model_candidate_dirs(app_file=app_file):
        existing = find_existing_model(base_dir)
        if existing:
            return existing
    return None


def resolve_model_path(model_path=None, app_file=None):
    if model_path:
        return str(model_path)
    existing = get_existing_model_path(app_file=app_file)
    if existing:
        return str(existing)
    return str(get_app_dir(app_file=app_file) / MODEL_FILENAME)
