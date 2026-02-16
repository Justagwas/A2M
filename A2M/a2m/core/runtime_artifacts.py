from __future__ import annotations

import os
import sys
from pathlib import Path

_CUDA_PROVIDER_NAMES = (
    'onnxruntime_providers_cuda.dll',
    'onnxruntime_providers_cuda.so',
    'onnxruntime_providers_cuda.dylib',
)
_DML_PROVIDER_NAMES = (
    'onnxruntime_providers_dml.dll',
    'DirectML.dll',
    'directml.dll',
)


def runtime_root_from_path(path: Path) -> Path:
    if not path.exists():
        return path
    if (path / 'onnxruntime').is_dir():
        return path
    try:
        for child in path.iterdir():
            if child.is_dir() and (child / 'onnxruntime').is_dir():
                return child
    except Exception:
        return path
    return path


def runtime_search_paths(runtime_root: Path) -> list[Path]:
    candidates = [
        runtime_root,
        runtime_root / 'bin',
        runtime_root / 'lib',
        runtime_root / 'onnxruntime' / 'capi',
    ]
    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            key = str(candidate.resolve()).lower()
        except Exception:
            key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(candidate)
    return resolved


def runtime_binary_exists(capi_dir: Path) -> bool:
    if os.name == 'nt':
        return (capi_dir / 'onnxruntime.dll').exists()
    if sys.platform == 'darwin':
        return any(((capi_dir / name).exists() for name in ('libonnxruntime.dylib', 'onnxruntime.dylib')))
    return any(((capi_dir / name).exists() for name in ('libonnxruntime.so', 'onnxruntime.so')))


def pybind_state_exists(capi_dir: Path) -> bool:
    patterns = ['onnxruntime_pybind11_state*.pyd'] if os.name == 'nt' else ['onnxruntime_pybind11_state*.so', 'onnxruntime_pybind11_state*.dylib']
    for pattern in patterns:
        if any(capi_dir.glob(pattern)):
            return True
    return False


def provider_artifact_exists(capi_dir: Path, provider: str) -> bool:
    normalized = str(provider or '').strip().lower()
    if normalized == 'cuda':
        names = _CUDA_PROVIDER_NAMES
    elif normalized == 'dml':
        names = _DML_PROVIDER_NAMES
    else:
        names = tuple()
    return any(((capi_dir / name).exists() for name in names))


def runtime_root_has_provider(path: Path, provider: str) -> bool:
    root = runtime_root_from_path(path)
    capi = root / 'onnxruntime' / 'capi'
    if not capi.is_dir():
        return False
    return provider_artifact_exists(capi, provider)


def runtime_root_looks_valid(path: Path) -> bool:
    root = runtime_root_from_path(path)
    package = root / 'onnxruntime'
    capi = package / 'capi'
    if not package.is_dir() or not capi.is_dir():
        return False
    if not runtime_binary_exists(capi):
        return False
    return pybind_state_exists(capi)
