from __future__ import annotations

import json
import uuid
from pathlib import Path
from threading import Event

from . import gpu_runtime_service, runtime_service
from .archive_service import assert_not_stopped as _assert_not_stopped
from .archive_service import remove_tree as _remove_tree
from .archive_service import safe_extract_zip as _safe_extract_zip
from .config import ONNX_CUDA_RUNTIME_PACK_URL, ONNX_DML_RUNTIME_PACK_URL
from .model_service import download_file
from .paths import localappdata_dir
from .runtime_artifacts import runtime_root_from_path as _shared_runtime_root_from_path

_PROVIDERS = ('cuda', 'dml')


def _normalize_provider(provider: str | None, *, allow_auto: bool=False) -> str:
    normalized = str(provider or '').strip().lower()
    allowed = set(_PROVIDERS)
    if allow_auto:
        allowed.add('auto')
    if normalized in allowed:
        return normalized
    return 'auto' if allow_auto else ''


def provider_display_name(provider: str | None) -> str:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return 'CUDA'
    if normalized == 'dml':
        return 'DirectML'
    return 'GPU'


def resolve_provider_for_preference(preference: str | None) -> str:
    normalized = _normalize_provider(preference, allow_auto=True)
    if normalized in _PROVIDERS:
        return normalized
    return gpu_runtime_service.resolve_provider_for_install('auto')


def runtime_packs_root() -> Path:
    root = localappdata_dir() / 'A2M' / 'runtime_packs'
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_updates_dir() -> Path:
    updates = localappdata_dir() / 'A2M' / 'updates'
    updates.mkdir(parents=True, exist_ok=True)
    return updates


def provider_pack_dir(provider: str) -> Path:
    normalized = _normalize_provider(provider)
    if not normalized:
        raise ValueError("provider must be 'cuda' or 'dml'")
    return runtime_packs_root() / normalized


def provider_pack_url(provider: str) -> str:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return str(ONNX_CUDA_RUNTIME_PACK_URL or '').strip()
    if normalized == 'dml':
        return str(ONNX_DML_RUNTIME_PACK_URL or '').strip()
    return ''


def _runtime_root_from_path(path: Path) -> Path:
    return _shared_runtime_root_from_path(path)


def _read_metadata(path: Path) -> dict[str, object]:
    metadata_path = path / 'runtime_metadata.json'
    if not metadata_path.exists():
        raise RuntimeError('Runtime pack metadata is missing.')
    try:
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise RuntimeError(f'Runtime pack metadata is invalid: {exc}') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('Runtime pack metadata is invalid.')
    return payload


def validate_pack_root(path: Path, expected_provider: str) -> None:
    provider = _normalize_provider(expected_provider)
    if not provider:
        raise RuntimeError("Expected provider must be 'cuda' or 'dml'.")
    runtime_root = _runtime_root_from_path(Path(path))
    package_dir = runtime_root / 'onnxruntime'
    capi_dir = package_dir / 'capi'
    if not package_dir.is_dir() or not capi_dir.is_dir():
        raise RuntimeError('Runtime pack is missing onnxruntime/capi payload.')
    metadata = _read_metadata(runtime_root)
    metadata_provider = str(metadata.get('provider', '') or '').strip().lower()
    if metadata_provider != provider:
        raise RuntimeError(f'Runtime pack provider mismatch: expected {provider}, got {metadata_provider or "unknown"}.')
    if not runtime_service.is_runtime_path_valid(runtime_root):
        raise RuntimeError('Runtime pack validation failed (ONNX runtime payload is incomplete).')
    detected_provider = gpu_runtime_service.detect_runtime_provider(runtime_root)
    if detected_provider != provider:
        raise RuntimeError(f'Runtime pack does not expose required {provider_display_name(provider)} provider.')


def _installed_pack_for_provider(provider: str) -> str:
    try:
        pack_dir = provider_pack_dir(provider)
    except Exception:
        return ''
    if not pack_dir.exists():
        return ''
    try:
        validate_pack_root(pack_dir, provider)
    except Exception:
        return ''
    return str(pack_dir)


def resolve_installed_pack(provider_pref: str) -> tuple[str, str]:
    normalized_pref = _normalize_provider(provider_pref, allow_auto=True)
    if normalized_pref in _PROVIDERS:
        candidates = [normalized_pref]
    else:
        preferred = resolve_provider_for_preference('auto')
        fallback = 'dml' if preferred == 'cuda' else 'cuda'
        candidates = [preferred, fallback]
    for provider in candidates:
        runtime_path = _installed_pack_for_provider(provider)
        if runtime_path:
            return (provider, runtime_path)
    return ('', '')


def download_and_install_pack(provider: str, *, progress_callback=None, stop_event: Event | None=None) -> Path:
    normalized_provider = _normalize_provider(provider)
    if not normalized_provider:
        raise RuntimeError("Provider must be 'cuda' or 'dml'.")
    pack_url = provider_pack_url(normalized_provider)
    if not pack_url:
        raise RuntimeError(f'{provider_display_name(normalized_provider)} runtime pack URL is not configured.')
    token = uuid.uuid4().hex
    archive_path = runtime_updates_dir() / f'a2m-onnx-{normalized_provider}-{token}.zip'
    stage_dir = runtime_packs_root() / f'.stage-{normalized_provider}-{token}'
    final_dir = provider_pack_dir(normalized_provider)
    backup_dir = runtime_packs_root() / f'.backup-{normalized_provider}-{token}'
    replaced_existing = False
    try:
        _assert_not_stopped(stop_event, message='Runtime pack download stopped by user.')
        download_file(pack_url, archive_path, progress_callback=progress_callback, stop_event=stop_event)
        _assert_not_stopped(stop_event, message='Runtime pack download stopped by user.')
        _safe_extract_zip(
            archive_path,
            stage_dir,
            stop_event=stop_event,
            stop_message='Runtime pack download stopped by user.',
            unsafe_entry_prefix='Unsafe runtime pack entry rejected',
            empty_message='Runtime pack archive is empty.',
        )
        runtime_root = _runtime_root_from_path(stage_dir)
        validate_pack_root(runtime_root, normalized_provider)
        if backup_dir.exists():
            _remove_tree(backup_dir)
        if final_dir.exists():
            final_dir.replace(backup_dir)
            replaced_existing = True
        runtime_root.replace(final_dir)
        if replaced_existing and backup_dir.exists():
            _remove_tree(backup_dir)
        return final_dir
    except Exception:
        if replaced_existing and backup_dir.exists() and (not final_dir.exists()):
            try:
                backup_dir.replace(final_dir)
            except Exception:
                pass
        raise
    finally:
        if archive_path.exists():
            try:
                archive_path.unlink()
            except Exception:
                pass
        if stage_dir.exists():
            _remove_tree(stage_dir)

