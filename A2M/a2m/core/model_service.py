from __future__ import annotations
import uuid
from pathlib import Path
from threading import Event
from typing import Callable
from .constants import DOWNLOAD_HEADER_PROFILES, DOWNLOAD_RETRIES_PER_HEADER, DOWNLOAD_RETRY_BACKOFF_SECONDS, DOWNLOAD_TIMEOUT_SECONDS, MODEL_FILENAME, MODEL_MIN_BYTES, MODEL_URL
from .http_service import download_file_with_retries
from .paths import app_dir, dedupe_paths, localappdata_dir
ProgressCallback = Callable[[float], None]
LogCallback = Callable[[str], None]
MODEL_DIR = localappdata_dir() / 'A2M' / 'models'

def get_model_candidate_dirs() -> list[Path]:
    primary = MODEL_DIR
    return dedupe_paths((primary, app_dir()))

def _is_valid_model_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and (path.stat().st_size >= MODEL_MIN_BYTES)
    except Exception:
        return False

def get_existing_model_path() -> Path | None:
    for base_dir in get_model_candidate_dirs():
        candidate = Path(base_dir) / MODEL_FILENAME
        if _is_valid_model_file(candidate):
            return candidate
    return None

def can_write_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_path = path / f'.a2m_write_test_{uuid.uuid4().hex}'
        with open(test_path, 'wb') as handle:
            handle.write(b'')
        test_path.unlink()
        return True
    except Exception:
        return False

def download_file(url: str, dest_path: Path | str, progress_callback: ProgressCallback | None=None, stop_event: Event | None=None) -> None:
    download_file_with_retries(
        url,
        dest_path,
        headers_profiles=DOWNLOAD_HEADER_PROFILES,
        timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
        retries_per_header=DOWNLOAD_RETRIES_PER_HEADER,
        retry_backoff_seconds=DOWNLOAD_RETRY_BACKOFF_SECONDS,
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_message='Model download stopped by user.',
        error_prefix='Failed to download model',
    )

def ensure_model_file(progress_callback: ProgressCallback | None=None, log_callback: LogCallback | None=None, stop_event: Event | None=None) -> Path:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError('Model download stopped by user.')
    existing_model = get_existing_model_path()
    if existing_model:
        return Path(existing_model)
    attempted_dirs: list[str] = []
    candidate_dirs = get_model_candidate_dirs()
    for index, target_dir in enumerate(candidate_dirs):
        target_dir = Path(target_dir)
        if not can_write_dir(target_dir):
            attempted_dirs.append(f'{target_dir} (not writable)')
            if log_callback:
                if index == 0 and len(candidate_dirs) > 1:
                    log_callback(f'App folder not writable. Falling back to cache:\n{MODEL_DIR}')
                else:
                    log_callback(f'Skipping unwritable folder:\n{target_dir}')
            continue
        model_path = target_dir / MODEL_FILENAME
        try:
            if model_path.exists():
                model_path.unlink()
            if log_callback:
                log_callback(f'Downloading ONNX model to:\n{model_path}')
            download_file(MODEL_URL, model_path, progress_callback, stop_event=stop_event)
            if not _is_valid_model_file(model_path):
                raise RuntimeError('Downloaded model file is incomplete or corrupted.')
            return model_path
        except InterruptedError:
            if model_path.exists():
                model_path.unlink()
            raise
        except Exception as exc:
            attempted_dirs.append(f'{target_dir} ({exc})')
            if log_callback and index < len(candidate_dirs) - 1:
                log_callback(f'Download failed in:\n{target_dir}\nTrying fallback location...')
    if not attempted_dirs:
        raise RuntimeError('No writable folder is available for model download.')
    raise RuntimeError('Model download failed in all locations:\n' + '\n'.join(attempted_dirs))

def get_model_install_hint() -> Path:
    candidate_dirs = get_model_candidate_dirs()
    if not candidate_dirs:
        return MODEL_DIR
    primary = candidate_dirs[0]
    if can_write_dir(primary):
        return primary
    return MODEL_DIR
