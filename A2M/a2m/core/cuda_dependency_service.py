from __future__ import annotations

import ctypes
import os
import uuid
from pathlib import Path
from threading import Event
from typing import Callable

from .archive_service import assert_not_stopped as _assert_not_stopped
from .archive_service import remove_tree as _remove_tree
from .archive_service import safe_extract_zip as _safe_extract_zip
from .constants import CUDNN_DOWNLOAD_URL
from .model_service import download_file
from .paths import dedupe_paths, localappdata_dir, normalized_path_key

try:
    import winreg
except Exception:  # pragma: no cover - non-Windows fallback
    winreg = None  # type: ignore[assignment]

ProgressCallback = Callable[[float], None]
_CUDA_REQUIRED_DLLS = (
    'cudart64_12.dll',
    'cublasLt64_12.dll',
    'cudnn64_9.dll',
)


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


def install_cudnn_runtime(*, progress_callback: ProgressCallback | None=None, stop_event: Event | None=None) -> Path:
    url = str(CUDNN_DOWNLOAD_URL or '').strip()
    if not url:
        raise RuntimeError('cuDNN download URL is not configured.')
    token = uuid.uuid4().hex
    archive_path = _updates_dir() / f'cudnn-{token}.zip'
    stage_dir = _dependencies_dir() / f'.stage-cudnn-{token}'
    final_root = _dependencies_dir() / 'cudnn'
    backup_root = _dependencies_dir() / f'.backup-cudnn-{token}'
    replaced_existing = False
    try:
        _assert_not_stopped(stop_event, message='cuDNN installation was stopped.')
        download_file(url, archive_path, progress_callback=progress_callback, stop_event=stop_event)
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
        if backup_root.exists():
            _remove_tree(backup_root)
        if final_root.exists():
            final_root.replace(backup_root)
            replaced_existing = True
        payload_root.replace(final_root)
        if replaced_existing and backup_root.exists():
            _remove_tree(backup_root)
        return _locate_cudnn_bin(final_root)
    except Exception:
        if replaced_existing and backup_root.exists() and (not final_root.exists()):
            try:
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


def missing_required_cuda_dlls() -> list[str]:
    env_dirs = [Path(str(entry or '').strip()) for entry in os.environ.get('PATH', '').split(os.pathsep) if str(entry or '').strip()]
    search_dirs = dedupe_paths([*env_dirs, *discover_cuda_runtime_bin_dirs()])
    missing: list[str] = []
    for dll_name in _CUDA_REQUIRED_DLLS:
        found = False
        for base in search_dirs:
            try:
                if (base / dll_name).exists():
                    found = True
                    break
            except Exception:
                continue
        if not found:
            missing.append(dll_name)
    return missing


def _broadcast_environment_change() -> None:
    if os.name != 'nt':
        return
    hwnd_broadcast = 0xFFFF
    wm_settingchange = 0x001A
    smto_abortifhung = 0x0002
    try:
        ctypes.windll.user32.SendMessageTimeoutW(hwnd_broadcast, wm_settingchange, 0, 'Environment', smto_abortifhung, 2000, None)
    except Exception:
        return


def add_bin_to_user_path(bin_dir: Path | str) -> bool:
    if os.name != 'nt' or winreg is None:
        return False
    path = Path(bin_dir).resolve()
    if not path.exists():
        raise RuntimeError(f'cuDNN bin folder does not exist: {path}')
    target = str(path)
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment', 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
            try:
                existing, _value_type = winreg.QueryValueEx(key, 'Path')
                existing_path = str(existing or '')
            except FileNotFoundError:
                existing_path = ''
            entries = [entry for entry in existing_path.split(os.pathsep) if entry]
            normalized_entries = {entry.strip().lower() for entry in entries}
            if target.strip().lower() in normalized_entries:
                return False
            updated = existing_path + ((os.pathsep + target) if existing_path else target)
            winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, updated)
    except Exception as exc:
        raise RuntimeError(f'Failed to update user PATH: {exc}') from exc
    _broadcast_environment_change()
    return True


def install_cudnn_and_configure_path(*, progress_callback: ProgressCallback | None=None, stop_event: Event | None=None) -> tuple[Path, bool, bool]:
    bin_dir = install_cudnn_runtime(progress_callback=progress_callback, stop_event=stop_event)
    process_changed = add_bin_to_process_path(bin_dir)
    user_changed = add_bin_to_user_path(bin_dir)
    return (bin_dir, process_changed, user_changed)
