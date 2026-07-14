from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from threading import Event

from . import gpu_runtime_service, runtime_service
from .archive_service import assert_not_stopped as _assert_not_stopped
from .archive_service import remove_tree as _remove_tree
from .archive_service import safe_extract_zip as _safe_extract_zip
from .config import ONNX_CUDA_RUNTIME_PACK_DOWNLOAD_SIZE, ONNX_CUDA_RUNTIME_PACK_INSTALLED_SIZE, ONNX_CUDA_RUNTIME_PACK_MAX_BYTES, ONNX_CUDA_RUNTIME_PACK_ORT_VERSION, ONNX_CUDA_RUNTIME_PACK_SHA256, ONNX_CUDA_RUNTIME_PACK_URL, ONNX_DML_RUNTIME_PACK_DOWNLOAD_SIZE, ONNX_DML_RUNTIME_PACK_INSTALLED_SIZE, ONNX_DML_RUNTIME_PACK_MAX_BYTES, ONNX_DML_RUNTIME_PACK_ORT_VERSION, ONNX_DML_RUNTIME_PACK_SHA256, ONNX_DML_RUNTIME_PACK_URL
from .model_service import download_file, get_existing_model_path
from .paths import app_dir, localappdata_dir
from .runtime_artifacts import runtime_root_from_path as _shared_runtime_root_from_path

_PROVIDERS = ('cuda', 'dml')


def _normalize_provider(provider: str | None) -> str:
    normalized = str(provider or '').strip().lower()
    if normalized in _PROVIDERS:
        return normalized
    return ''


def provider_display_name(provider: str | None) -> str:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return 'CUDA'
    if normalized == 'dml':
        return 'DirectML'
    return 'GPU'


def resolve_provider_for_preference(preference: str | None) -> str:
    normalized = _normalize_provider(preference)
    return normalized if normalized else 'dml'


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


def provider_pack_download_size(provider: str) -> int:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return int(ONNX_CUDA_RUNTIME_PACK_DOWNLOAD_SIZE)
    if normalized == 'dml':
        return int(ONNX_DML_RUNTIME_PACK_DOWNLOAD_SIZE)
    return 0


def provider_pack_installed_size(provider: str) -> int:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return int(ONNX_CUDA_RUNTIME_PACK_INSTALLED_SIZE)
    if normalized == 'dml':
        return int(ONNX_DML_RUNTIME_PACK_INSTALLED_SIZE)
    return 0


def provider_pack_size_summary(provider: str) -> str:
    download_bytes = provider_pack_download_size(provider)
    installed_bytes = provider_pack_installed_size(provider)
    if download_bytes <= 0 or installed_bytes <= 0:
        return 'Install size is unavailable.'
    mib = 1024 * 1024
    download_mib = max(1, (download_bytes + mib - 1) // mib)
    installed_mib = max(1, (installed_bytes + mib - 1) // mib)
    return f'About {download_mib} MB to download.'


def provider_pack_sha256(provider: str) -> str:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return str(ONNX_CUDA_RUNTIME_PACK_SHA256 or '').strip()
    if normalized == 'dml':
        return str(ONNX_DML_RUNTIME_PACK_SHA256 or '').strip()
    return ''


def provider_pack_max_bytes(provider: str) -> int:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return int(ONNX_CUDA_RUNTIME_PACK_MAX_BYTES)
    if normalized == 'dml':
        return int(ONNX_DML_RUNTIME_PACK_MAX_BYTES)
    return 0


def provider_pack_ort_version(provider: str) -> str:
    normalized = _normalize_provider(provider)
    if normalized == 'cuda':
        return str(ONNX_CUDA_RUNTIME_PACK_ORT_VERSION or '').strip()
    if normalized == 'dml':
        return str(ONNX_DML_RUNTIME_PACK_ORT_VERSION or '').strip()
    return ''


def _require_provider_pack_sha256(provider: str) -> str:
    digest = provider_pack_sha256(provider)
    if re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None:
        raise RuntimeError(f'{provider_display_name(provider)} runtime pack integrity hash is not configured.')
    return digest.lower()


def _runtime_root_from_path(path: Path) -> Path:
    return _shared_runtime_root_from_path(path)


def _read_metadata(path: Path) -> dict[str, object]:
    metadata_path = path / 'runtime_metadata.json'
    if not metadata_path.exists():
        raise RuntimeError('Runtime pack metadata is missing.')
    try:
        payload = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        raise RuntimeError(f'Runtime pack metadata is invalid: {exc}') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('Runtime pack metadata is invalid.')
    return payload


def _helper_command(task: str, runtime_path: Path, provider: str, model_path: Path | None=None) -> list[str]:
    args = ['--gpu-helper', task, '--runtime-path', str(runtime_path), '--provider', provider]
    if model_path is not None:
        args.extend(['--model', str(model_path)])
    if getattr(sys, 'frozen', False):
        return [str(sys.executable), *args]
    return [str(sys.executable), str(app_dir() / 'A2M.py'), *args]


def validate_pack_in_helper(path: Path, expected_provider: str) -> None:
    provider = _normalize_provider(expected_provider)
    if not provider:
        raise RuntimeError("Expected provider must be 'cuda' or 'dml'.")
    runtime_root = _runtime_root_from_path(Path(path))
    model_path = get_existing_model_path()
    task = 'validate-provider' if provider == 'cuda' else ('create-session' if model_path is not None else 'validate-provider')
    cmd = _helper_command(task, runtime_root, provider, model_path=model_path)
    kwargs: dict[str, object] = {'capture_output': True, 'text': True, 'timeout': 45, 'check': False}
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs['startupinfo'] = startupinfo
        kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        proc = subprocess.run(cmd, **kwargs)
    except Exception as exc:
        raise RuntimeError(f'Runtime pack helper validation failed to run: {exc}') from exc
    payload: dict[str, object] = {}
    for line in reversed(str(proc.stdout or '').splitlines()):
        try:
            candidate = json.loads(line.strip())
        except Exception:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if int(proc.returncode) != 0 or not bool(payload.get('ok', False)):
        reason = str(payload.get('reason_text') or payload.get('details') or proc.stderr or 'Runtime pack failed helper validation.').strip()
        raise RuntimeError(reason)


def validate_pack_root(path: Path, expected_provider: str) -> None:
    provider = _normalize_provider(expected_provider)
    if not provider:
        raise RuntimeError("Expected provider must be 'cuda' or 'dml'.")
    runtime_root = _runtime_root_from_path(Path(path))
    package_dir = runtime_root / 'onnxruntime'
    capi_dir = package_dir / 'capi'
    if not package_dir.is_dir() or not (package_dir / '__init__.py').is_file() or not capi_dir.is_dir():
        raise RuntimeError('Runtime pack is missing onnxruntime/capi payload.')
    metadata = _read_metadata(runtime_root)
    metadata_provider = str(metadata.get('provider', '') or '').strip().lower()
    if metadata_provider != provider:
        raise RuntimeError(f'Runtime pack provider mismatch: expected {provider}, got {metadata_provider or "unknown"}.')
    expected_ort_version = provider_pack_ort_version(provider)
    metadata_ort_version = str(metadata.get('ort_version', '') or '').strip()
    if expected_ort_version and metadata_ort_version != expected_ort_version:
        raise RuntimeError(
            f'Runtime pack version mismatch: A2M requires ONNX Runtime {expected_ort_version}, '
            f'but the pack contains {metadata_ort_version or "an unknown version"}.'
        )
    python_tag = str(metadata.get('python_tag', '') or '').strip().lower()
    implementation_name = str(getattr(sys.implementation, 'name', '') or '').strip().lower()
    if implementation_name == 'cpython':
        expected_python_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    else:
        expected_python_tag = str(getattr(sys.implementation, 'cache_tag', '') or '').strip().lower()
    if python_tag and expected_python_tag and python_tag != expected_python_tag:
        raise RuntimeError(
            f'Runtime pack Python ABI mismatch: this app requires {expected_python_tag}, '
            f'but the pack contains {python_tag}.'
        )
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
    provider = resolve_provider_for_preference(provider_pref)
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
        expected_sha256 = _require_provider_pack_sha256(normalized_provider)
        download_file(
            pack_url,
            archive_path,
            progress_callback=progress_callback,
            stop_event=stop_event,
            expected_sha256=expected_sha256,
            max_download_bytes=provider_pack_max_bytes(normalized_provider),
            stop_message='Runtime pack download stopped by user.',
            error_prefix=f'Failed to download {provider_display_name(normalized_provider)} runtime pack',
        )
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
        validate_pack_in_helper(runtime_root, normalized_provider)
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
