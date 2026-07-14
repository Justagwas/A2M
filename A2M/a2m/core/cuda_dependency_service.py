from __future__ import annotations

import importlib.metadata
import contextlib
import io
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from .archive_service import assert_not_stopped as _assert_not_stopped
from .archive_service import remove_tree as _remove_tree
from .archive_service import safe_extract_zip as _safe_extract_zip
from .config import CUDNN_CUDA13_DOWNLOAD_SHA256, CUDNN_CUDA13_DOWNLOAD_SIZE, CUDNN_CUDA13_DOWNLOAD_URL
from .config import CUDNN_DOWNLOAD_MAX_BYTES, CUDNN_DOWNLOAD_SHA256, CUDNN_DOWNLOAD_SIZE, CUDNN_DOWNLOAD_URL
from .config import ONNX_CUDA_RUNTIME_PACK_ORT_VERSION
from .model_service import download_file
from .paths import dedupe_paths, localappdata_dir, normalized_path_key
from .runtime_artifacts import runtime_root_from_path

ProgressCallback = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class CudaRuntimeRequirements:
    ort_version: str
    cuda_majors: tuple[int, ...]
    cudnn_major: int


def _parse_version_parts(value: str | None) -> tuple[int, int, int]:
    raw = str(value or '').strip()
    parts: list[int] = []
    for token in raw.replace('-', '.').split('.'):
        if not token.isdigit():
            break
        parts.append(int(token))
        if len(parts) >= 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _runtime_metadata_version(runtime_path: Path | str | None) -> str:
    raw_path = str(runtime_path or os.environ.get('A2M_GPU_RUNTIME_PATH', '') or '').strip()
    if not raw_path:
        return ''
    try:
        root = runtime_root_from_path(Path(raw_path))
    except Exception:
        return ''
    metadata_path = root / 'runtime_metadata.json'
    if not metadata_path.exists():
        return ''
    try:
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
    except Exception:
        return ''
    if not isinstance(payload, dict):
        return ''
    provider = str(payload.get('provider', '') or '').strip().lower()
    if provider and provider != 'cuda':
        return ''
    return str(payload.get('ort_version', '') or '').strip()


def _installed_onnxruntime_gpu_version() -> str:
    for package_name in ('onnxruntime-gpu',):
        try:
            return str(importlib.metadata.version(package_name))
        except Exception:
            continue
    try:
        import onnxruntime as ort
        providers = tuple(str(provider) for provider in ort.get_available_providers())
        if 'CUDAExecutionProvider' not in providers:
            return ''
        return str(getattr(ort, '__version__', '') or '')
    except Exception:
        return ''


def get_cuda_runtime_requirements(runtime_path: Path | str | None=None) -> CudaRuntimeRequirements:
    version = _runtime_metadata_version(runtime_path) or _installed_onnxruntime_gpu_version() or str(ONNX_CUDA_RUNTIME_PACK_ORT_VERSION or '').strip()
    major, minor, patch = _parse_version_parts(version)
    if (major, minor) >= (1, 27):
        return CudaRuntimeRequirements(ort_version=version, cuda_majors=(13,), cudnn_major=9)
    if (major, minor, patch) >= (1, 18, 1):
        return CudaRuntimeRequirements(ort_version=version, cuda_majors=(12,), cudnn_major=9)
    if (major, minor) >= (1, 17):
        return CudaRuntimeRequirements(ort_version=version, cuda_majors=(12,), cudnn_major=8)
    return CudaRuntimeRequirements(ort_version=version, cuda_majors=(11,), cudnn_major=8)


def cuda_requirements_summary(runtime_path: Path | str | None=None) -> str:
    requirements = get_cuda_runtime_requirements(runtime_path)
    cuda_text = ' or '.join((f'CUDA {major}.x' for major in requirements.cuda_majors))
    ort_text = f' for onnxruntime-gpu {requirements.ort_version}' if requirements.ort_version else ''
    return f'{cuda_text} and cuDNN {requirements.cudnn_major}.x{ort_text}'


def required_cuda_dll_groups(runtime_path: Path | str | None=None) -> tuple[tuple[str, ...], ...]:
    requirements = get_cuda_runtime_requirements(runtime_path)
    cuda_majors = tuple(int(major) for major in requirements.cuda_majors)
    return (
        tuple((f'cudart64_{major}.dll' for major in cuda_majors)),
        tuple((f'cublasLt64_{major}.dll' for major in cuda_majors)),
        (f'cudnn64_{requirements.cudnn_major}.dll',),
    )


def _cudnn_download_for_cuda_major(cuda_major: int) -> tuple[str, str, int, int]:
    major = int(cuda_major)
    if major >= 13:
        return (
            CUDNN_CUDA13_DOWNLOAD_URL,
            CUDNN_CUDA13_DOWNLOAD_SHA256,
            CUDNN_CUDA13_DOWNLOAD_SIZE,
            CUDNN_DOWNLOAD_MAX_BYTES,
        )
    return (CUDNN_DOWNLOAD_URL, CUDNN_DOWNLOAD_SHA256, CUDNN_DOWNLOAD_SIZE, CUDNN_DOWNLOAD_MAX_BYTES)


def selected_cudnn_download(runtime_path: Path | str | None=None) -> tuple[str, str, int, int]:
    requirements = get_cuda_runtime_requirements(runtime_path)
    cuda_major = max(requirements.cuda_majors) if requirements.cuda_majors else 12
    return _cudnn_download_for_cuda_major(int(cuda_major))


def _updates_dir() -> Path:
    path = localappdata_dir() / 'A2M' / 'updates'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dependencies_dir() -> Path:
    path = localappdata_dir() / 'A2M' / 'dependencies'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _windows_cuda_bin_dirs() -> list[Path]:
    if os.name != 'nt':
        return []
    candidates: list[Path] = []
    env_names = ['CUDA_PATH'] + [f'CUDA_PATH_V{major}_{minor}' for major in range(9, 15) for minor in range(0, 10)]
    for name in env_names:
        raw = str(os.environ.get(name, '') or '').strip()
        if not raw:
            continue
        base = Path(raw)
        candidates.append(base / 'bin')
    cuda_root = Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA')
    if cuda_root.exists():
        for child in sorted(cuda_root.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            child_name = child.name.lower()
            if not child_name.startswith('v'):
                continue
            candidates.append(child / 'bin')
    return dedupe_paths(candidates)


def _windows_cudnn_bin_dirs() -> list[Path]:
    if os.name != 'nt':
        return []
    candidates: list[Path] = []
    cudnn_root = Path(r'C:\Program Files\NVIDIA\CUDNN')
    if cudnn_root.exists():
        for child in sorted(cudnn_root.iterdir(), reverse=True):
            if child.is_dir():
                candidates.append(child / 'bin')
    local_dependency_cudnn = _dependencies_dir() / 'cudnn'
    if local_dependency_cudnn.exists():
        for match in local_dependency_cudnn.rglob('cudnn64_9.dll'):
            try:
                if match.is_file():
                    candidates.append(match.parent)
            except Exception:
                continue
    return dedupe_paths(candidates)


def discover_cuda_runtime_bin_dirs() -> list[Path]:
    candidates = _windows_cuda_bin_dirs() + _windows_cudnn_bin_dirs()
    return dedupe_paths(candidates, existing_only=True)


def ensure_cuda_runtime_bins_in_process_path() -> list[Path]:
    added: list[Path] = []
    for bin_dir in discover_cuda_runtime_bin_dirs():
        try:
            changed = add_bin_to_process_path(bin_dir)
        except Exception:
            continue
        if changed:
            added.append(Path(bin_dir))
    return added


def preload_onnxruntime_cuda_dlls() -> bool:
    try:
        import onnxruntime as ort
    except Exception:
        return False
    preload = getattr(ort, 'preload_dlls', None)
    if not callable(preload):
        return False
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            preload(cuda=True, cudnn=True, msvc=True)
        return True
    except Exception:
        return False


def _payload_root(stage_dir: Path) -> Path:
    entries = [entry for entry in stage_dir.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return stage_dir


def _locate_cudnn_bin(path: Path) -> Path:
    matches = [candidate.parent for candidate in path.rglob('cudnn64_9.dll') if candidate.is_file()]
    if not matches:
        raise RuntimeError('cuDNN package does not contain cudnn64_9.dll.')
    # Prefer the shallowest match when multiple bins are present.
    return sorted(matches, key=lambda p: len(p.parts))[0]


def install_cudnn_runtime(*, progress_callback: ProgressCallback | None=None, stop_event: Event | None=None, runtime_path: Path | str | None=None) -> Path:
    url, expected_sha256, expected_size, max_bytes = selected_cudnn_download(runtime_path)
    url = str(url or '').strip()
    if not url:
        raise RuntimeError('cuDNN download URL is not configured.')
    expected_sha256 = str(expected_sha256 or '').strip()
    if re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha256) is None:
        raise RuntimeError('cuDNN download integrity hash is not configured or invalid.')
    token = uuid.uuid4().hex
    archive_path = _updates_dir() / f'cudnn-{token}.zip'
    stage_dir = _dependencies_dir() / f'.stage-cudnn-{token}'
    final_root = _dependencies_dir() / 'cudnn'
    backup_root = _dependencies_dir() / f'.backup-cudnn-{token}'
    replaced_existing = False
    try:
        _assert_not_stopped(stop_event, message='cuDNN installation was stopped.')
        download_file(
            url,
            archive_path,
            progress_callback=progress_callback,
            stop_event=stop_event,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            max_download_bytes=max_bytes,
            stop_message='cuDNN installation was stopped.',
            error_prefix='Failed to download cuDNN',
        )
        _assert_not_stopped(stop_event, message='cuDNN installation was stopped.')
        _safe_extract_zip(
            archive_path,
            stage_dir,
            stop_event=stop_event,
            stop_message='cuDNN installation was stopped.',
            unsafe_entry_prefix='Unsafe cuDNN archive entry rejected',
            empty_message='cuDNN archive is empty.',
        )
        payload_root = _payload_root(stage_dir)
        _locate_cudnn_bin(payload_root)
        if backup_root.exists():
            _remove_tree(backup_root)
        if final_root.exists():
            final_root.replace(backup_root)
            replaced_existing = True
        payload_root.replace(final_root)
        installed_bin = _locate_cudnn_bin(final_root)
        if replaced_existing and backup_root.exists():
            _remove_tree(backup_root)
        return installed_bin
    except Exception:
        if replaced_existing and backup_root.exists():
            try:
                if final_root.exists():
                    _remove_tree(final_root)
                backup_root.replace(final_root)
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


def add_bin_to_process_path(bin_dir: Path | str) -> bool:
    path = Path(bin_dir).resolve()
    if not path.exists():
        raise RuntimeError(f'cuDNN bin folder does not exist: {path}')
    target = str(path)
    entries = [entry for entry in os.environ.get('PATH', '').split(os.pathsep) if entry]
    normalized = {normalized_path_key(entry) for entry in entries}
    if normalized_path_key(path) in normalized:
        return False
    os.environ['PATH'] = target + (os.pathsep + os.environ['PATH'] if os.environ.get('PATH') else '')
    return True


def missing_required_cuda_dlls(*, runtime_path: Path | str | None=None) -> list[str]:
    env_dirs = [Path(str(entry or '').strip()) for entry in os.environ.get('PATH', '').split(os.pathsep) if str(entry or '').strip()]
    search_dirs = dedupe_paths([*env_dirs, *discover_cuda_runtime_bin_dirs()])
    missing: list[str] = []
    for dll_group in required_cuda_dll_groups(runtime_path):
        found = False
        for base in search_dirs:
            for dll_name in dll_group:
                try:
                    if (base / dll_name).exists():
                        found = True
                        break
                except Exception:
                    continue
            if found:
                break
        if not found:
            missing.append(' or '.join(dll_group))
    return missing


def install_cudnn_and_configure_path(*, progress_callback: ProgressCallback | None=None, stop_event: Event | None=None, runtime_path: Path | str | None=None) -> tuple[Path, bool, bool]:
    bin_dir = install_cudnn_runtime(progress_callback=progress_callback, stop_event=stop_event, runtime_path=runtime_path)
    process_changed = add_bin_to_process_path(bin_dir)
    return (bin_dir, process_changed, False)
