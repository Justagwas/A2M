from __future__ import annotations
import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .runtime_artifacts import provider_artifact_exists as _shared_provider_artifact_exists
from .runtime_artifacts import pybind_state_exists as _shared_pybind_state_exists
from .runtime_artifacts import runtime_binary_exists as _shared_runtime_binary_exists
from .runtime_path_service import prime_runtime_path as _shared_prime_runtime_path

@dataclass(slots=True)
class RuntimeProbe:
    runtime_available: bool
    cuda_available: bool
    dml_available: bool
    available_providers: tuple[str, ...]
    error: str = ''
_ONNXRUNTIME_MODULE: Any | None = None
_ONNXRUNTIME_RUNTIME_KEY = ''
_PROBE_CACHE: dict[str, RuntimeProbe] = {}
_RUNTIME_SYS_PATHS: set[str] = set()
_RUNTIME_PATH_PREFIXES: set[str] = set()
_DLL_DIR_HANDLES: dict[str, object] = {}
_CPU_PROVIDER = 'CPUExecutionProvider'
_CUDA_PROVIDER = 'CUDAExecutionProvider'
_DML_PROVIDER = 'DmlExecutionProvider'

def _normalize_runtime_key(runtime_path: str | Path | None) -> str:
    if not runtime_path:
        return ''
    try:
        return str(Path(runtime_path).resolve())
    except Exception:
        return str(runtime_path)

def _clear_onnxruntime_modules() -> None:
    for name in [n for n in list(sys.modules) if n == 'onnxruntime' or n.startswith('onnxruntime.')]:
        sys.modules.pop(name, None)

def _release_dll_dir_handles() -> None:
    for handle in list(_DLL_DIR_HANDLES.values()):
        try:
            close = getattr(handle, 'close', None)
            if callable(close):
                close()
        except Exception:
            pass
    _DLL_DIR_HANDLES.clear()

def _remove_runtime_sys_paths() -> None:
    if not _RUNTIME_SYS_PATHS:
        return
    keep = [entry for entry in sys.path if entry not in _RUNTIME_SYS_PATHS]
    sys.path[:] = keep
    _RUNTIME_SYS_PATHS.clear()


def _remove_runtime_path_prefixes() -> None:
    if not _RUNTIME_PATH_PREFIXES:
        return
    existing = [p for p in os.environ.get('PATH', '').split(os.pathsep) if p]
    blocked = {str(Path(p)).lower() for p in _RUNTIME_PATH_PREFIXES}
    filtered = []
    for entry in existing:
        try:
            normalized = str(Path(entry)).lower()
        except Exception:
            normalized = str(entry).lower()
        if normalized in blocked:
            continue
        filtered.append(entry)
    os.environ['PATH'] = os.pathsep.join(filtered)
    _RUNTIME_PATH_PREFIXES.clear()

def _prime_runtime_path(runtime_path: Path) -> None:
    _shared_prime_runtime_path(
        runtime_path,
        tracked_sys_paths=_RUNTIME_SYS_PATHS,
        tracked_env_prefixes=_RUNTIME_PATH_PREFIXES,
        track_existing_env_prefixes=True,
        dll_dir_handles=_DLL_DIR_HANDLES,
    )

def _import_runtime_package_from_path(runtime_root: Path):
    package_dir = runtime_root / 'onnxruntime'
    capi_dir = package_dir / 'capi'
    init_file = package_dir / '__init__.py'
    if not package_dir.is_dir():
        raise RuntimeError('ONNX runtime package not found in runtime path.')
    if not init_file.is_file():
        if capi_dir.is_dir():
            return importlib.import_module('onnxruntime')
        raise RuntimeError('ONNX runtime package not found in runtime path.')
    spec = importlib.util.spec_from_file_location('onnxruntime', str(init_file), submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        raise RuntimeError('Failed to load ONNX runtime package spec from runtime path.')
    module = importlib.util.module_from_spec(spec)
    sys.modules['onnxruntime'] = module
    spec.loader.exec_module(module)
    return module

def _import_onnxruntime(*, runtime_path: str | Path | None=None, force_reload: bool=False):
    global _ONNXRUNTIME_MODULE, _ONNXRUNTIME_RUNTIME_KEY
    runtime_key = _normalize_runtime_key(runtime_path)
    if _ONNXRUNTIME_MODULE is not None and runtime_key != _ONNXRUNTIME_RUNTIME_KEY:
        raise RuntimeError('ONNX runtime backend changed in this session. Please restart A2M and try again.')
    needs_reload = force_reload or _ONNXRUNTIME_MODULE is None or runtime_key != _ONNXRUNTIME_RUNTIME_KEY
    if not needs_reload:
        return _ONNXRUNTIME_MODULE
    _clear_onnxruntime_modules()
    _release_dll_dir_handles()
    if runtime_key:
        runtime_root = Path(runtime_key)
        _prime_runtime_path(runtime_root)
        _ONNXRUNTIME_MODULE = _import_runtime_package_from_path(runtime_root)
    else:
        _ONNXRUNTIME_MODULE = importlib.import_module('onnxruntime')
    _ONNXRUNTIME_RUNTIME_KEY = runtime_key
    return _ONNXRUNTIME_MODULE

def reset_runtime_cache(*, clear_import_cache: bool=False) -> None:
    global _ONNXRUNTIME_MODULE, _ONNXRUNTIME_RUNTIME_KEY
    _PROBE_CACHE.clear()
    if clear_import_cache:
        _clear_onnxruntime_modules()
        _release_dll_dir_handles()
        _remove_runtime_sys_paths()
        _remove_runtime_path_prefixes()
        _ONNXRUNTIME_MODULE = None
        _ONNXRUNTIME_RUNTIME_KEY = ''

def _provider_dll_exists(capi_dir: Path, provider: str) -> bool:
    return _shared_provider_artifact_exists(capi_dir, provider)


def _runtime_binary_exists(capi_dir: Path) -> bool:
    return _shared_runtime_binary_exists(capi_dir)


def _pybind_state_exists(capi_dir: Path) -> bool:
    return _shared_pybind_state_exists(capi_dir)

def _probe_from_runtime_path(runtime_key: str) -> RuntimeProbe:
    runtime_root = Path(runtime_key)
    package_dir = runtime_root / 'onnxruntime'
    if not package_dir.is_dir():
        return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error='ONNX runtime package not found in runtime path.')
    capi_dir = package_dir / 'capi'
    if not capi_dir.is_dir():
        return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error='ONNX runtime package is missing capi directory.')
    if not _runtime_binary_exists(capi_dir):
        return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error='ONNX runtime package is missing core runtime binary.')
    if not _pybind_state_exists(capi_dir):
        return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error='ONNX runtime package is missing pybind state binary.')
    cuda_available = _provider_dll_exists(capi_dir, 'cuda')
    dml_available = _provider_dll_exists(capi_dir, 'dml')
    providers: list[str] = []
    if cuda_available:
        providers.append(_CUDA_PROVIDER)
    if dml_available:
        providers.append(_DML_PROVIDER)
    providers.append(_CPU_PROVIDER)
    return RuntimeProbe(runtime_available=True, cuda_available=cuda_available, dml_available=dml_available, available_providers=tuple(providers), error='')

def probe_runtime_path(runtime_path: str | Path) -> RuntimeProbe:
    runtime_key = _normalize_runtime_key(runtime_path)
    if not runtime_key:
        return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error='Runtime path is empty.')
    return _probe_from_runtime_path(runtime_key)

def _providers_from_module(module: Any, *, default_cpu: bool) -> tuple[str, ...]:
    providers = tuple((str(p) for p in module.get_available_providers()))
    if default_cpu and not providers:
        return (_CPU_PROVIDER,)
    return providers

def _runtime_probe_from_providers(providers: tuple[str, ...], *, error: str='') -> RuntimeProbe:
    return RuntimeProbe(
        runtime_available=True,
        cuda_available=_CUDA_PROVIDER in providers,
        dml_available=_DML_PROVIDER in providers,
        available_providers=providers,
        error=error,
    )

def _probe_from_environment() -> RuntimeProbe:
    spec = importlib.util.find_spec('onnxruntime')
    if spec is None:
        return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error='onnxruntime package not found.')
    if _ONNXRUNTIME_MODULE is None or _ONNXRUNTIME_RUNTIME_KEY:
        try:
            ort = _import_onnxruntime(runtime_path=None, force_reload=False)
            providers = _providers_from_module(ort, default_cpu=True)
            return _runtime_probe_from_providers(providers)
        except Exception as exc:
            return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error=str(exc))
    try:
        providers = _providers_from_module(_ONNXRUNTIME_MODULE, default_cpu=True)
        return _runtime_probe_from_providers(providers)
    except Exception as exc:
        return RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error=str(exc))

def probe_runtime_support(*, force_refresh: bool=False, runtime_path: str | Path | None=None) -> RuntimeProbe:
    runtime_key = _normalize_runtime_key(runtime_path)
    if not force_refresh and runtime_key in _PROBE_CACHE:
        return _PROBE_CACHE[runtime_key]
    loaded_matches_runtime = _ONNXRUNTIME_MODULE is not None and runtime_key == _ONNXRUNTIME_RUNTIME_KEY
    if loaded_matches_runtime:
        try:
            providers = _providers_from_module(_ONNXRUNTIME_MODULE, default_cpu=False)
            probe = _runtime_probe_from_providers(providers)
        except Exception as exc:
            probe = RuntimeProbe(runtime_available=False, cuda_available=False, dml_available=False, available_providers=tuple(), error=str(exc))
    elif runtime_key:
        probe = _probe_from_runtime_path(runtime_key)
    else:
        probe = _probe_from_environment()
    _PROBE_CACHE[runtime_key] = probe
    return probe

def resolve_provider_order(*, device_preference: str, gpu_provider_preference: str, probe: RuntimeProbe) -> list[str]:
    if not probe.runtime_available:
        return [_CPU_PROVIDER]
    preference = str(device_preference or 'cpu').strip().lower()
    gpu_pref = str(gpu_provider_preference or 'auto').strip().lower()
    providers = set(probe.available_providers)
    if preference != 'gpu':
        return [_CPU_PROVIDER]
    chosen: list[str] = []
    if gpu_pref == 'cuda':
        chosen.append(_CUDA_PROVIDER)
    elif gpu_pref == 'dml':
        chosen.append(_DML_PROVIDER)
    elif _CUDA_PROVIDER in providers:
        chosen.append(_CUDA_PROVIDER)
    elif _DML_PROVIDER in providers:
        chosen.append(_DML_PROVIDER)
    chosen = [p for p in chosen if p in providers]
    if _CPU_PROVIDER in providers or not chosen:
        chosen.append(_CPU_PROVIDER)
    return chosen

def provider_display_name(provider_name: str | None) -> str:
    normalized = str(provider_name or '').strip()
    if normalized == _CUDA_PROVIDER:
        return 'CUDA'
    if normalized == _DML_PROVIDER:
        return 'DirectML'
    return 'CPU'

def create_session(*, model_path: Path | str, device_preference: str, gpu_provider_preference: str, runtime_path: str | Path | None=None, force_refresh: bool=False):
    probe = probe_runtime_support(force_refresh=force_refresh, runtime_path=runtime_path)
    if not probe.runtime_available:
        detail = f' ({probe.error})' if probe.error else ''
        raise RuntimeError(f'ONNX Runtime is not available{detail}.')
    ort = _import_onnxruntime(runtime_path=runtime_path, force_reload=False)
    provider_order = resolve_provider_order(device_preference=device_preference, gpu_provider_preference=gpu_provider_preference, probe=probe)
    requested_gpu = str(device_preference or '').strip().lower() == 'gpu'
    if requested_gpu and (not provider_order or provider_order[0] == _CPU_PROVIDER):
        preferred = str(gpu_provider_preference or 'auto').strip().lower()
        if preferred == 'cuda':
            raise RuntimeError('GPU mode selected, but CUDA provider is not available.')
        if preferred == 'dml':
            raise RuntimeError('GPU mode selected, but DirectML provider is not available.')
        raise RuntimeError('GPU mode selected, but no GPU provider is available (CUDA/DirectML missing).')
    options = ort.SessionOptions()
    primary_provider = provider_order[0] if provider_order else _CPU_PROVIDER
    using_dml = primary_provider == _DML_PROVIDER
    options.enable_mem_pattern = False if using_dml else True
    if using_dml:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_path = str(Path(model_path))
    try:
        session = ort.InferenceSession(session_path, sess_options=options, providers=provider_order)
    except Exception as exc:
        if requested_gpu:
            raise RuntimeError(f'GPU mode selected, but ONNX GPU session could not be created: {exc}') from exc
        session = ort.InferenceSession(session_path, sess_options=options, providers=[_CPU_PROVIDER])
    active_provider = _CPU_PROVIDER
    try:
        providers = session.get_providers()
        if providers:
            active_provider = str(providers[0])
    except Exception:
        active_provider = _CPU_PROVIDER
    if requested_gpu and active_provider == _CPU_PROVIDER:
        raise RuntimeError('GPU mode selected, but ONNX session started on CPU.')
    return (session, active_provider, probe)
