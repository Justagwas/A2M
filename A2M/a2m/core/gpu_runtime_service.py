from __future__ import annotations
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from .runtime_artifacts import provider_artifact_exists as _shared_provider_artifact_exists
from .runtime_artifacts import pybind_state_exists as _shared_pybind_state_exists
from .runtime_artifacts import runtime_binary_exists as _shared_runtime_binary_exists
from .runtime_artifacts import runtime_root_from_path as _shared_runtime_root_from_path


def _run_hidden_process(args: list[str], timeout_seconds: int=1) -> subprocess.CompletedProcess[str] | None:
    kwargs: dict[str, object] = {
        'capture_output': True,
        'text': True,
        'timeout': timeout_seconds,
        'check': False,
    }
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs['startupinfo'] = startupinfo
        kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        return subprocess.run(args, **kwargs)
    except Exception:
        return None


def _has_nvidia_gpu_via_display_devices() -> bool:
    if os.name != 'nt':
        return False
    try:
        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_uint32),
                ('DeviceName', ctypes.c_wchar * 32),
                ('DeviceString', ctypes.c_wchar * 128),
                ('StateFlags', ctypes.c_uint32),
                ('DeviceID', ctypes.c_wchar * 128),
                ('DeviceKey', ctypes.c_wchar * 128),
            ]
        enum_fn = ctypes.windll.user32.EnumDisplayDevicesW
        index = 0
        while True:
            display = DISPLAY_DEVICEW()
            display.cb = ctypes.sizeof(DISPLAY_DEVICEW)
            if not enum_fn(None, index, ctypes.byref(display), 0):
                break
            name = f'{display.DeviceName} {display.DeviceString} {display.DeviceID}'.upper()
            if 'NVIDIA' in name:
                return True
            index += 1
    except Exception:
        return False
    return False


def _has_nvidia_gpu() -> bool:
    if _has_nvidia_gpu_via_display_devices():
        return True
    result = _run_hidden_process(['nvidia-smi', '-L'])
    if result is None:
        return False
    output = (result.stdout or '') + '\n' + (result.stderr or '')
    return result.returncode == 0 and 'GPU' in output.upper()


def _get_nvidia_gpu_name_from_smi() -> str:
    result = _run_hidden_process(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], timeout_seconds=2)
    if result is None or result.returncode != 0:
        return ''
    for line in (result.stdout or '').splitlines():
        candidate = str(line).strip()
        if candidate:
            return candidate
    return ''


def _list_display_gpu_names() -> list[str]:
    if os.name != 'nt':
        return []
    names: list[str] = []
    seen: set[str] = set()
    try:
        class DISPLAY_DEVICEW(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_uint32),
                ('DeviceName', ctypes.c_wchar * 32),
                ('DeviceString', ctypes.c_wchar * 128),
                ('StateFlags', ctypes.c_uint32),
                ('DeviceID', ctypes.c_wchar * 128),
                ('DeviceKey', ctypes.c_wchar * 128),
            ]
        enum_fn = ctypes.windll.user32.EnumDisplayDevicesW
        index = 0
        while True:
            display = DISPLAY_DEVICEW()
            display.cb = ctypes.sizeof(DISPLAY_DEVICEW)
            if not enum_fn(None, index, ctypes.byref(display), 0):
                break
            raw_name = str(display.DeviceString or '').strip()
            if not raw_name:
                index += 1
                continue
            upper_name = raw_name.upper()
            if 'MICROSOFT BASIC RENDER DRIVER' in upper_name:
                index += 1
                continue
            if upper_name not in seen:
                seen.add(upper_name)
                names.append(raw_name)
            index += 1
    except Exception:
        return []
    return names


def resolve_provider_for_install(preference: str | None='auto') -> str:
    normalized = str(preference or 'auto').strip().lower()
    if normalized in {'cuda', 'dml'}:
        return normalized
    return 'cuda' if _has_nvidia_gpu() else 'dml'


def get_gpu_model_name(provider_preference: str | None='auto') -> str:
    normalized = str(provider_preference or 'auto').strip().lower()
    if normalized not in {'auto', 'cuda', 'dml'}:
        normalized = 'auto'
    if normalized in {'auto', 'cuda'}:
        nvidia_name = _get_nvidia_gpu_name_from_smi()
        if nvidia_name:
            return nvidia_name
    names = _list_display_gpu_names()
    if not names:
        return ''
    if normalized in {'auto', 'cuda'}:
        for name in names:
            if 'NVIDIA' in name.upper():
                return name
    return names[0]


def _provider_runtime_artifacts_present(runtime_root: Path, provider: str) -> bool:
    capi_dir = runtime_root / 'onnxruntime' / 'capi'
    if not capi_dir.is_dir():
        return False
    return _shared_provider_artifact_exists(capi_dir, provider)


def _runtime_binary_present(capi_dir: Path) -> bool:
    return _shared_runtime_binary_exists(capi_dir)


def _pybind_state_present(capi_dir: Path) -> bool:
    return _shared_pybind_state_exists(capi_dir)


def _read_runtime_metadata(runtime_path: Path) -> dict[str, object]:
    candidates = [runtime_path / 'runtime_metadata.json', runtime_path.parent / 'runtime_metadata.json']
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _resolve_runtime_package_root(target_dir: Path) -> Path:
    return _shared_runtime_root_from_path(target_dir)


def _looks_like_runtime_root(path: Path, provider: str) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    package_dir = path / 'onnxruntime'
    capi_dir = package_dir / 'capi'
    if not package_dir.is_dir() or not capi_dir.is_dir():
        return False
    if not _runtime_binary_present(capi_dir):
        return False
    if not _pybind_state_present(capi_dir):
        return False
    return _provider_runtime_artifacts_present(path, provider)


def detect_runtime_provider(runtime_path: str | Path | None) -> str:
    if not runtime_path:
        return ''
    try:
        root = _resolve_runtime_package_root(Path(runtime_path))
    except Exception:
        return ''
    metadata = _read_runtime_metadata(root)
    provider = str(metadata.get('provider', '') or '').strip().lower()
    if provider in {'cuda', 'dml'}:
        return provider
    if _looks_like_runtime_root(root, 'cuda'):
        return 'cuda'
    if _looks_like_runtime_root(root, 'dml'):
        return 'dml'
    return ''
