from __future__ import annotations
import json
import re
import uuid
from pathlib import Path
from threading import Event
from typing import Callable
from .archive_service import remove_tree, safe_extract_zip
from .config import DOWNLOAD_HEADER_PROFILES, DOWNLOAD_RETRIES_PER_HEADER, DOWNLOAD_RETRY_BACKOFF_SECONDS, DOWNLOAD_TIMEOUT_SECONDS, MODEL_DOWNLOAD_MAX_BYTES, MODEL_FILENAME, MODEL_ID, MODEL_INSTALL_DIRNAME, MODEL_MIN_BYTES, MODEL_SCHEMA_VERSION, MODEL_SHA256, MODEL_SIZE_BYTES, MODEL_URL, MODEL_VERSION
from .http_service import DownloadValidationError, download_file_with_retries, sha256_file
from .paths import app_dir, dedupe_paths, localappdata_dir, normalized_path_key
ProgressCallback = Callable[[float], None]
LogCallback = Callable[[str], None]
MODEL_DIR = localappdata_dir() / 'A2M' / 'models'
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_MODEL_HASH_CACHE: dict[tuple[str, int, int, str], bool] = {}
_REQUIRED_COMPONENTS = ('scorer.onnx', 'attributes.onnx', 'frontend.npz')
def get_model_candidate_dirs() -> list[Path]:
    primary = MODEL_DIR
    return dedupe_paths((primary, app_dir()))

def _expected_model_sha256() -> str:
    candidate = str(MODEL_SHA256 or '').strip().lower()
    if _SHA256_RE.fullmatch(candidate):
        return candidate
    return ''

def _model_hash_cache_key(path: Path, stat_result: object, expected_sha256: str) -> tuple[str, int, int, str]:
    path_key = normalized_path_key(path)
    size = int(getattr(stat_result, 'st_size', 0))
    mtime_ns = int(getattr(stat_result, 'st_mtime_ns', 0))
    return path_key, size, mtime_ns, expected_sha256

def _is_valid_model_file(path: Path, *, expected_sha256: str | None=None, minimum_bytes: int=1) -> bool:
    try:
        if not path.exists() or not path.is_file():
            return False
        stat_result = path.stat()
        if stat_result.st_size < max(1, int(minimum_bytes)):
            return False
        expected_digest = str(expected_sha256 or '').strip().lower()
        if not _SHA256_RE.fullmatch(expected_digest):
            return True
        cache_key = _model_hash_cache_key(path, stat_result, expected_digest)
        cached = _MODEL_HASH_CACHE.get(cache_key)
        if cached is not None:
            return cached
        valid = sha256_file(path) == expected_digest
        path_key = cache_key[0]
        for stale_key in tuple(_MODEL_HASH_CACHE):
            if stale_key[0] == path_key and stale_key != cache_key:
                _MODEL_HASH_CACHE.pop(stale_key, None)
        if len(_MODEL_HASH_CACHE) >= 32:
            _MODEL_HASH_CACHE.pop(next(iter(_MODEL_HASH_CACHE)), None)
        _MODEL_HASH_CACHE[cache_key] = valid
        return valid
    except Exception:
        return False

def _bundle_install_dir(base_dir: Path) -> Path:
    return Path(base_dir) / MODEL_INSTALL_DIRNAME


def _read_manifest(bundle_dir: Path) -> dict | None:
    try:
        payload = json.loads((bundle_dir / 'manifest.json').read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _validate_installed_bundle(bundle_dir: Path) -> bool:
    manifest = _read_manifest(bundle_dir)
    if not manifest:
        return False
    try:
        schema_version = int(manifest.get('schema_version', -1))
    except (TypeError, ValueError, OverflowError):
        return False
    if schema_version != int(MODEL_SCHEMA_VERSION):
        return False
    if str(manifest.get('model_id', '')).strip() != str(MODEL_ID):
        return False
    if str(manifest.get('model_version', '')).strip() != str(MODEL_VERSION):
        return False
    components = manifest.get('components')
    if not isinstance(components, dict):
        return False
    for name in _REQUIRED_COMPONENTS:
        record = components.get(name)
        if not isinstance(record, dict):
            return False
        try:
            expected_size = int(record.get('bytes', 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return False
        expected_hash = str(record.get('sha256', '') or '').strip().lower()
        component_path = bundle_dir / name
        if expected_size <= 0 or not _SHA256_RE.fullmatch(expected_hash):
            return False
        if not _is_valid_model_file(component_path, expected_sha256=expected_hash):
            return False
        try:
            if component_path.stat().st_size != expected_size:
                return False
        except Exception:
            return False
    return True


def get_existing_model_bundle_dir() -> Path | None:
    for base_dir in get_model_candidate_dirs():
        candidate = _bundle_install_dir(Path(base_dir))
        if _validate_installed_bundle(candidate):
            remove_tree(candidate.parent / f'.{candidate.name}.backup')
            return candidate
        backup = candidate.parent / f'.{candidate.name}.backup'
        if _validate_installed_bundle(backup):
            try:
                remove_tree(candidate)
                backup.replace(candidate)
                return candidate
            except OSError:
                return backup
        remove_tree(backup)
    return None


def get_existing_model_path() -> Path | None:
    bundle_dir = get_existing_model_bundle_dir()
    return (bundle_dir / 'scorer.onnx') if bundle_dir is not None else None

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

def download_file(url: str, dest_path: Path | str, progress_callback: ProgressCallback | None=None, stop_event: Event | None=None, *, expected_sha256: str | None=None, expected_size: int | None=None, max_download_bytes: int | None=None, stop_message: str='Model download stopped by user.', error_prefix: str='Failed to download model') -> None:
    download_file_with_retries(
        url,
        dest_path,
        headers_profiles=DOWNLOAD_HEADER_PROFILES,
        timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
        retries_per_header=DOWNLOAD_RETRIES_PER_HEADER,
        retry_backoff_seconds=DOWNLOAD_RETRY_BACKOFF_SECONDS,
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_message=stop_message,
        error_prefix=error_prefix,
        expected_sha256=MODEL_SHA256 if expected_sha256 is None else expected_sha256,
        expected_size=MODEL_SIZE_BYTES if expected_size is None and str(url or '').strip() == str(MODEL_URL) else expected_size,
        max_download_bytes=MODEL_DOWNLOAD_MAX_BYTES if max_download_bytes is None else max_download_bytes,
    )

def _install_downloaded_bundle(archive_path: Path, target_dir: Path, *, stop_event: Event | None=None) -> Path:
    staging_dir = target_dir.parent / f'.{target_dir.name}.install-{uuid.uuid4().hex}'
    backup_dir = target_dir.parent / f'.{target_dir.name}.backup'
    remove_tree(staging_dir)
    if backup_dir.exists():
        if target_dir.exists():
            remove_tree(backup_dir)
        else:
            backup_dir.replace(target_dir)
    installed = False
    try:
        safe_extract_zip(
            archive_path,
            staging_dir,
            stop_event=stop_event,
            stop_message='Model installation stopped by user.',
            max_total_uncompressed_bytes=128 * 1024 * 1024,
        )
        if not _validate_installed_bundle(staging_dir):
            raise RuntimeError('Downloaded model package contents are incomplete or corrupted.')
        if target_dir.exists():
            target_dir.replace(backup_dir)
        try:
            staging_dir.replace(target_dir)
        except Exception:
            if backup_dir.exists() and not target_dir.exists():
                try:
                    backup_dir.replace(target_dir)
                except Exception as restore_error:
                    raise RuntimeError(
                        f'Model installation failed and the previous model could not be restored. '
                        f'The backup remains at: {backup_dir}'
                    ) from restore_error
            raise
        installed = True
        remove_tree(backup_dir)
        return target_dir / 'scorer.onnx'
    finally:
        remove_tree(staging_dir)
        if installed:
            remove_tree(backup_dir)


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
                    log_callback(f'Primary model folder is not writable. Trying the fallback location:\n{candidate_dirs[1]}')
                else:
                    log_callback(f'Skipping unwritable folder:\n{target_dir}')
            continue
        model_path = target_dir / MODEL_FILENAME
        install_dir = _bundle_install_dir(target_dir)
        try:
            if model_path.exists():
                model_path.unlink()
            if log_callback:
                log_callback(f'Downloading Transcription Model package to:\n{model_path}')
            download_file(MODEL_URL, model_path, progress_callback, stop_event=stop_event)
            if not _is_valid_model_file(model_path, expected_sha256=_expected_model_sha256(), minimum_bytes=MODEL_MIN_BYTES):
                raise RuntimeError('Downloaded model package is incomplete or corrupted.')
            scorer_path = _install_downloaded_bundle(model_path, install_dir, stop_event=stop_event)
            model_path.unlink(missing_ok=True)
            return scorer_path
        except InterruptedError:
            if model_path.exists():
                model_path.unlink()
            raise
        except DownloadValidationError:
            try:
                model_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        except Exception as exc:
            try:
                model_path.unlink(missing_ok=True)
            except Exception:
                pass
            attempted_dirs.append(f'{target_dir} ({exc})')
            if log_callback and index < len(candidate_dirs) - 1:
                log_callback(f'Download failed in:\n{target_dir}\nTrying fallback location...')
    if not attempted_dirs:
        raise RuntimeError('No writable folder is available for model download.')
    raise RuntimeError('Model download failed in all locations:\n' + '\n'.join(attempted_dirs))

def get_model_install_hint() -> Path:
    candidate_dirs = get_model_candidate_dirs()
    if not candidate_dirs:
        return MODEL_DIR / MODEL_INSTALL_DIRNAME
    primary = candidate_dirs[0]
    if can_write_dir(primary):
        return _bundle_install_dir(primary)
    return MODEL_DIR / MODEL_INSTALL_DIRNAME
