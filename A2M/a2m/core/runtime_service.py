from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path
from .constants import GPU_BATCH_SIZE_MAX, GPU_BATCH_SIZE_MIN
from .onnx_runtime_service import RuntimeProbe, create_session as create_onnx_session
from .onnx_runtime_service import probe_runtime_path as probe_runtime_path_only
from .onnx_runtime_service import probe_runtime_support as probe_onnx_runtime_support
from .onnx_runtime_service import provider_display_name, reset_runtime_cache as reset_onnx_cache
from .paths import app_dir, bundle_dir
from .runtime_artifacts import runtime_root_from_path as _shared_runtime_root_from_path
from .runtime_artifacts import runtime_root_has_provider as _shared_runtime_root_has_provider
from .runtime_artifacts import runtime_root_looks_valid as _shared_runtime_root_looks_valid
_DEVICE_PREFERENCE = 'cpu'
_GPU_BATCH_SIZE = 4
_GPU_PROVIDER_PREFERENCE = 'auto'
_GPU_RUNTIME_ENABLED = False
_GPU_RUNTIME_PATH = ''


def _runtime_root_from_path(path: Path) -> Path:
    return _shared_runtime_root_from_path(path)


def _runtime_root_looks_valid(path: Path) -> bool:
    return _shared_runtime_root_looks_valid(path)


def _candidate_key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except Exception:
        return str(path).lower()


def _packaged_runtime_candidates() -> list[Path]:
    candidates: list[Path] = [
        app_dir(),
        app_dir() / '_internal',
        bundle_dir(),
        bundle_dir() / '_internal',
    ]
    if not getattr(sys, 'frozen', False):
        spec = importlib.util.find_spec('onnxruntime')
        if spec is not None:
            locations = list(spec.submodule_search_locations or [])
            if locations:
                for location in locations:
                    candidates.append(Path(location).parent)
            elif spec.origin:
                candidates.append(Path(str(spec.origin)).parent.parent)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _runtime_root_has_cuda_provider(path: Path) -> bool:
    return _shared_runtime_root_has_provider(path, 'cuda')


def get_packaged_runtime_root() -> str:
    for candidate in _packaged_runtime_candidates():
        if _runtime_root_looks_valid(candidate):
            try:
                return str(_runtime_root_from_path(candidate).resolve())
            except Exception:
                return str(_runtime_root_from_path(candidate))
    return ''


def is_packaged_runtime_available() -> bool:
    runtime_root = get_packaged_runtime_root()
    if not runtime_root:
        return False
    return _runtime_root_looks_valid(Path(runtime_root))


def is_gpu_capable_build() -> bool:
    runtime_root = get_packaged_runtime_root()
    if not runtime_root:
        return False
    return _runtime_root_has_cuda_provider(Path(runtime_root))


def get_effective_runtime_root_for_gpu_checks() -> str:
    if _GPU_RUNTIME_ENABLED and _GPU_RUNTIME_PATH:
        return _GPU_RUNTIME_PATH
    return get_packaged_runtime_root()


def is_runtime_path_valid(path: str | Path | None) -> bool:
    candidate = str(path or '').strip()
    if not candidate:
        return False
    try:
        probe = probe_runtime_path_only(candidate)
    except Exception:
        return False
    return bool(probe.runtime_available)


def resolve_working_runtime_path(preferred_path: str | Path | None=None) -> str:
    preferred = str(preferred_path or '').strip()
    if preferred and is_runtime_path_valid(preferred):
        try:
            return str(_runtime_root_from_path(Path(preferred)).resolve())
        except Exception:
            return str(_runtime_root_from_path(Path(preferred)))
    packaged = get_packaged_runtime_root()
    if packaged and is_runtime_path_valid(packaged):
        return packaged
    return ''


def _normalize_device_preference(preference: str | None) -> str:
    normalized = str(preference or 'cpu').strip().lower()
    if normalized not in {'cpu', 'gpu'}:
        normalized = 'cpu'
    return normalized


def _normalize_gpu_provider_preference(preference: str | None) -> str:
    normalized = str(preference or 'auto').strip().lower()
    if normalized in {'auto', 'cuda', 'dml'}:
        return normalized
    return 'auto'


def set_device_preference(preference: str | None) -> str:
    global _DEVICE_PREFERENCE
    normalized = _normalize_device_preference(preference)
    _DEVICE_PREFERENCE = normalized
    os.environ['A2M_DEVICE'] = normalized
    return _DEVICE_PREFERENCE


def get_device_preference() -> str:
    return _DEVICE_PREFERENCE


def set_gpu_batch_size(value: int | str | None) -> int:
    global _GPU_BATCH_SIZE
    try:
        batch = int(value)
    except Exception:
        batch = GPU_BATCH_SIZE_MIN
    batch = max(GPU_BATCH_SIZE_MIN, min(GPU_BATCH_SIZE_MAX, batch))
    _GPU_BATCH_SIZE = batch
    os.environ['A2M_BATCH_SIZE'] = str(batch)
    return batch


def get_gpu_batch_size() -> int:
    return _GPU_BATCH_SIZE


def set_gpu_provider_preference(preference: str | None) -> str:
    global _GPU_PROVIDER_PREFERENCE
    normalized = _normalize_gpu_provider_preference(preference)
    _GPU_PROVIDER_PREFERENCE = normalized
    return _GPU_PROVIDER_PREFERENCE


def get_gpu_provider_preference() -> str:
    return _GPU_PROVIDER_PREFERENCE


def set_gpu_runtime(enabled: bool | None, runtime_path: str | Path | None) -> tuple[bool, str]:
    global _GPU_RUNTIME_ENABLED, _GPU_RUNTIME_PATH
    selected_path = ''
    requested = bool(enabled)
    if requested:
        selected_path = resolve_working_runtime_path(runtime_path)
        requested = bool(selected_path)
    requested_path = selected_path if requested else ''
    if bool(_GPU_RUNTIME_ENABLED) == bool(requested) and str(_GPU_RUNTIME_PATH or '').strip() == str(requested_path or '').strip():
        return (_GPU_RUNTIME_ENABLED, _GPU_RUNTIME_PATH)
    _GPU_RUNTIME_ENABLED = requested
    _GPU_RUNTIME_PATH = requested_path
    if _GPU_RUNTIME_ENABLED:
        os.environ['A2M_GPU_RUNTIME_PATH'] = _GPU_RUNTIME_PATH
    else:
        os.environ.pop('A2M_GPU_RUNTIME_PATH', None)
    reset_runtime_cache(clear_import_cache=True)
    return (_GPU_RUNTIME_ENABLED, _GPU_RUNTIME_PATH)


def is_gpu_runtime_enabled() -> bool:
    return _GPU_RUNTIME_ENABLED


def get_gpu_runtime_path() -> str:
    return _GPU_RUNTIME_PATH


def _runtime_path_for_probe() -> str | None:
    if _GPU_RUNTIME_ENABLED and _GPU_RUNTIME_PATH:
        return _GPU_RUNTIME_PATH
    return None


def get_runtime_probe(*, force_refresh: bool=False) -> RuntimeProbe:
    return probe_onnx_runtime_support(force_refresh=force_refresh, runtime_path=_runtime_path_for_probe())


def is_runtime_available(*, force_refresh: bool=False) -> bool:
    return get_runtime_probe(force_refresh=force_refresh).runtime_available


def is_cuda_available(*, force_refresh: bool=False) -> bool:
    return get_runtime_probe(force_refresh=force_refresh).cuda_available


def is_dml_available(*, force_refresh: bool=False) -> bool:
    return get_runtime_probe(force_refresh=force_refresh).dml_available


def probe_runtime_support(*, force_refresh: bool=False) -> tuple[bool, bool, bool]:
    probe = get_runtime_probe(force_refresh=force_refresh)
    return (probe.runtime_available, probe.cuda_available, probe.dml_available)


def create_session(model_path: Path | str, *, force_refresh: bool=False, device_preference: str | None=None, gpu_provider_preference: str | None=None):
    effective_device = get_device_preference() if device_preference is None else _normalize_device_preference(device_preference)
    effective_gpu_provider = get_gpu_provider_preference() if gpu_provider_preference is None else _normalize_gpu_provider_preference(gpu_provider_preference)
    session, active_provider, _probe = create_onnx_session(model_path=model_path, device_preference=effective_device, gpu_provider_preference=effective_gpu_provider, runtime_path=_runtime_path_for_probe(), force_refresh=force_refresh)
    return (session, provider_display_name(active_provider))


def reset_runtime_cache(*, clear_import_cache: bool=True) -> None:
    reset_onnx_cache(clear_import_cache=clear_import_cache)
