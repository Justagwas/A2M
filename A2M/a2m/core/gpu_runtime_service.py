from __future__ import annotations
import ctypes
import json
import os
import shutil
import subprocess
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


def _nvidia_smi_path() -> str:
    if os.name != 'nt':
        return str(shutil.which('nvidia-smi') or '')
    candidates: list[Path] = []
    windows_dir = str(os.environ.get('WINDIR') or '').strip()
    if windows_dir:
        candidates.append(Path(windows_dir) / 'System32' / 'nvidia-smi.exe')
    for env_name in ('ProgramW6432', 'ProgramFiles'):
        program_files = str(os.environ.get(env_name) or '').strip()
        if program_files:
            candidates.append(Path(program_files) / 'NVIDIA Corporation' / 'NVSMI' / 'nvidia-smi.exe')
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except OSError:
            continue
    return ''


def _get_nvidia_gpu_name_from_smi() -> str:
    nvidia_smi = _nvidia_smi_path()
    if not nvidia_smi:
        return ''
    result = _run_hidden_process([nvidia_smi, '--query-gpu=name', '--format=csv,noheader'], timeout_seconds=2)
    if result is None or result.returncode != 0:
        return ''
    for line in (result.stdout or '').splitlines():
        candidate = str(line).strip()
        if candidate:
            return candidate
    return ''


def get_gpu_memory_mib(provider_preference: str | None = 'dml') -> int:
    normalized = str(provider_preference or 'dml').strip().lower()
    if normalized not in {'cuda', 'dml'}:
        normalized = 'dml'
    nvidia_memory = _get_nvidia_gpu_memory()
    if normalized == 'cuda':
        return nvidia_memory[0][1] if nvidia_memory else 0

    adapters = _list_display_adapters()
    if adapters:
        adapter_name, device_key = adapters[0]
        matched_memory = _match_nvidia_memory(adapter_name, nvidia_memory)
        if matched_memory > 0:
            return matched_memory
        if 'NVIDIA' in adapter_name.upper() and nvidia_memory and all(not name for name, _ in nvidia_memory):
            return max(memory for _, memory in nvidia_memory)
        registry_memory = _display_adapter_memory_mib(device_key)
        if registry_memory > 0:
            return registry_memory

    if not nvidia_memory:
        return 0
    named_memory = [memory for name, memory in nvidia_memory if name]
    if named_memory:
        return min(named_memory)
    return max((memory for _, memory in nvidia_memory), default=0)


def _get_nvidia_gpu_memory() -> list[tuple[str, int]]:
    nvidia_smi = _nvidia_smi_path()
    if not nvidia_smi:
        return []
    result = _run_hidden_process(
        [nvidia_smi, '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        timeout_seconds=2,
    )
    detected: list[tuple[str, int]] = []
    if result is not None and result.returncode == 0:
        for line in (result.stdout or '').splitlines():
            name, separator, raw_memory = str(line).strip().rpartition(',')
            if not separator:
                continue
            try:
                memory = int(raw_memory.strip().split()[0])
            except Exception:
                continue
            if name.strip() and memory > 0:
                detected.append((name.strip(), memory))
    if detected:
        return detected

    result = _run_hidden_process(
        [nvidia_smi, '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
        timeout_seconds=2,
    )
    if result is None or result.returncode != 0:
        return []
    for line in (result.stdout or '').splitlines():
        try:
            memory = int(str(line).strip().split()[0])
        except Exception:
            continue
        if memory > 0:
            detected.append(('', memory))
    return detected


def _normalized_adapter_name(value: str) -> str:
    return ''.join(character.lower() for character in str(value or '') if character.isalnum())


def _match_nvidia_memory(adapter_name: str, detected: list[tuple[str, int]]) -> int:
    target = _normalized_adapter_name(adapter_name)
    if not target:
        return 0
    for name, memory in detected:
        candidate = _normalized_adapter_name(name)
        if candidate and (candidate in target or target in candidate):
            return max(0, int(memory))
    return 0


def _list_display_adapters() -> list[tuple[str, str]]:
    if os.name != 'nt':
        return []
    adapters: list[tuple[str, str]] = []
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
                adapters.append((raw_name, str(display.DeviceKey or '').strip()))
            index += 1
    except Exception:
        return []
    return adapters


def _list_display_gpu_names() -> list[str]:
    return [name for name, _ in _list_display_adapters()]


def _display_adapter_memory_mib(device_key: str) -> int:
    if os.name != 'nt':
        return 0
    normalized_key = str(device_key or '').replace('/', '\\').strip()
    prefix = '\\Registry\\Machine\\'
    if not normalized_key.lower().startswith(prefix.lower()):
        return 0
    registry_path = normalized_key[len(prefix):]
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, registry_path) as key:
            for value_name in ('HardwareInformation.qwMemorySize', 'HardwareInformation.MemorySize'):
                try:
                    value, _value_type = winreg.QueryValueEx(key, value_name)
                except OSError:
                    continue
                if isinstance(value, bytes):
                    memory_bytes = int.from_bytes(value[:8], byteorder='little', signed=False)
                else:
                    memory_bytes = int(value)
                memory_mib = memory_bytes // (1024 * 1024)
                if 16 <= memory_mib <= 1024 * 1024:
                    return int(memory_mib)
    except Exception:
        return 0
    return 0


def get_gpu_model_name(provider_preference: str | None='dml') -> str:
    normalized = str(provider_preference or 'dml').strip().lower()
    if normalized not in {'cuda', 'dml'}:
        normalized = 'dml'
    if normalized == 'cuda':
        nvidia_name = _get_nvidia_gpu_name_from_smi()
        if nvidia_name:
            return nvidia_name
    names = _list_display_gpu_names()
    if not names:
        return ''
    if normalized == 'cuda':
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
